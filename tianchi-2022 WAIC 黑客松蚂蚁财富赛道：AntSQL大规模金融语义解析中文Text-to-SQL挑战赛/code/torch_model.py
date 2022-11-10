import time
import torch
import os
import numpy as np
import transformers
import utils
from torch import nn
from featurizer import SQLDataset


class HydraTorch(object):
    def __init__(self, config):
        self.config = config
        self.model = HydraNet(config)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer, self.scheduler = None, None

    def train_on_batch(self, batch):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": float(self.config["decay"]),
                },
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=float(self.config["learning_rate"]))
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(self.config["num_warmup_steps"]),
                num_training_steps=int(self.config["num_train_steps"]))
            self.optimizer.zero_grad()

        self.model.train()

        for k, v in batch.items():
            batch[k] = v.to(self.device)

        loss = self.model(**batch)["loss"]
        batch_loss = torch.mean(loss)
        batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return batch_loss.cpu().detach().numpy()

    def model_inference(self, model_inputs):
        self.model.eval()
        model_outputs = {}
        batch_size = 512
        for start_idx in range(0, model_inputs["input_ids"].shape[0], batch_size):
            input_tensor = {k: torch.from_numpy(model_inputs[k][start_idx:start_idx+batch_size]).to(self.device) for k in ["input_ids", "input_mask", "segment_ids"]}
            with torch.no_grad():
                model_output = self.model(**input_tensor)
            for k, out_tensor in model_output.items():
                if out_tensor is None:
                    continue
                if k not in model_outputs:
                    model_outputs[k] = []
                model_outputs[k].append(out_tensor.cpu().detach().numpy())

        for k in model_outputs:
            model_outputs[k] = np.concatenate(model_outputs[k], 0)

        return model_outputs

    def save(self, model_path, epoch):
        if "SAVE" in self.config and "DEBUG" not in self.config:
            save_path = os.path.join(model_path, "model_{0}.pt".format(epoch))
            if torch.cuda.device_count() > 1:
                torch.save(self.model.module.state_dict(), save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
            print("Model saved in path: %s" % save_path)

    def load(self, model_path, epoch):
        pt_path = os.path.join(model_path, "model_{0}.pt".format(epoch))
        loaded_dict = torch.load(pt_path, map_location=torch.device(self.device))
        if torch.cuda.device_count() > 1:
            self.model.module.load_state_dict(loaded_dict)
        else:
            self.model.load_state_dict(loaded_dict)
        print("PyTorch model loaded from {0}".format(pt_path))

    def dataset_inference(self, dataset: SQLDataset):
        print("model prediction start")
        start_time = time.time()
        model_outputs = self.model_inference(dataset.model_inputs)

        final_outputs = []
        for pos in dataset.pos:
            final_output = {}
            for k in model_outputs:
                final_output[k] = model_outputs[k][pos[0]:pos[1], :]
            final_outputs.append(final_output)
        print("model prediction end, time elapse: {0}".format(time.time() - start_time))
        assert len(dataset.input_features) == len(final_outputs)

        return final_outputs

    def predict_SQL(self, dataset: SQLDataset, model_outputs=None):
        if model_outputs is None:
            model_outputs = self.dataset_inference(dataset)
        sqls = []
        for input_feature, model_output in zip(dataset.input_features, model_outputs):
            select, where, conditions, wherenum = self.parse_output(input_feature, model_output, [])

            conditions_with_value_texts = []
            for wc in where:
                _, op, vs, ve = conditions[wc]
                value_span_text = "".join(input_feature.tokens[wc][vs:ve])
                value_span_text = value_span_text.replace('#', "")
                conditions_with_value_texts.append((wc, op, value_span_text))

            sqls.append((select, conditions_with_value_texts, wherenum))

        return sqls


    def parse_output(self, input_feature, model_output, where_label = []):
        def get_span(i):
            offset = 0
            segment_ids = np.array(input_feature.segment_ids[i])
            for j in range(len(segment_ids)):
                if segment_ids[j] == 1:
                    offset = j
                    break

            value_start, value_end = model_output["value_start"][i, segment_ids == 1], model_output["value_end"][i, segment_ids == 1]
            l = len(value_start)
            sum_mat = value_start.reshape((l, 1)) + value_end.reshape((1, l))
            span = (0, 0)
            for cur_span, _ in sorted(np.ndenumerate(sum_mat), key=lambda x:x[1], reverse=True):
                if cur_span[1] < cur_span[0] or cur_span[0] > l - 1 or cur_span[1] > l - 1:
                    continue
                span = cur_span
                break

            return (span[0]+offset, span[1]+offset)

        select_id_prob = sorted(enumerate(model_output["column_func"][:, 0]), key=lambda x:x[1], reverse=True)
        select = select_id_prob[0][0]

        where_id_prob = sorted(enumerate(model_output["column_func"][:, 1]), key=lambda x:x[1], reverse=True)
        relevant_prob = 1 - np.exp(model_output["column_func"][:, 2])
        where_num_scores = np.average(model_output["where_num"], axis=0, weights=relevant_prob)
        where_num = int(np.argmax(where_num_scores))
        where = [i for i, _ in where_id_prob[:where_num]]
        conditions = {}
        for idx in set(where + where_label):
            span = get_span(idx)
            op = np.argmax(model_output["op"][idx, :])
            conditions[idx] = (idx, op, span[0], span[1])

        return select, where, conditions, where_num

class HydraNet(nn.Module):
    def __init__(self, config):
        super(HydraNet, self).__init__()
        self.config = config
        self.base_model = utils.create_base_model(config)

        drop_rate = float(config["drop_rate"]) if "drop_rate" in config else 0.0
        self.dropout = nn.Dropout(drop_rate)

        bert_hid_size = self.base_model.config.hidden_size
        self.column_func = nn.Linear(bert_hid_size, 3)
        self.op = nn.Linear(bert_hid_size, int(config["op_num"]))
        self.where_num = nn.Linear(bert_hid_size, int(config["where_column_num"]) + 1)
        self.start_end = nn.Linear(bert_hid_size, 2)

    def forward(self, input_ids, input_mask, segment_ids, select=None, where=None, where_num=None, op=None, value_start=None, value_end=None):

        bert_output, pooled_output = self.base_model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            return_dict=False)

        bert_output = self.dropout(bert_output)
        pooled_output = self.dropout(pooled_output)

        column_func_logit = self.column_func(pooled_output)
        op_logit = self.op(pooled_output)
        where_num_logit = self.where_num(pooled_output)
        start_end_logit = self.start_end(bert_output)
        value_span_mask = input_mask.to(dtype=bert_output.dtype)

        start_logit = start_end_logit[:, :, 0] * value_span_mask - 1000000.0 * (1 - value_span_mask)
        end_logit = start_end_logit[:, :, 1] * value_span_mask - 1000000.0 * (1 - value_span_mask)

        loss = None
        if select is not None:
            bceloss = nn.BCEWithLogitsLoss(reduction="none")
            cross_entropy = nn.CrossEntropyLoss(reduction="none")

            loss = bceloss(column_func_logit[:, 0], select.float())
            loss += bceloss(column_func_logit[:, 1], where.float())
            loss += bceloss(column_func_logit[:, 2], (1-select.float()) * (1-where.float()))
            loss += cross_entropy(where_num_logit, where_num)
            loss += cross_entropy(op_logit, op) * where.float()
            loss += cross_entropy(start_logit, value_start)
            loss += cross_entropy(end_logit, value_end)

        log_sigmoid = nn.LogSigmoid()

        return {"column_func": log_sigmoid(column_func_logit),
                "op": op_logit.log_softmax(1),
                "where_num": where_num_logit.log_softmax(1),
                "value_start": start_logit.log_softmax(1),
                "value_end": end_logit.log_softmax(1),
                "loss": loss}

