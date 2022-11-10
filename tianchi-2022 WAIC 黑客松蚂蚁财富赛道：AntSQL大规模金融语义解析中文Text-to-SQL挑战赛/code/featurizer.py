import numpy as np
import utils
import torch.utils.data as torch_data
from sqlexample import SQLExample
from collections import defaultdict
import json

stats = defaultdict(int)

def reverse_pad(text, bantoken):

    reverse_dict = {
        '六个月':['这半年里', '近半年'],
        '中风险':['中等风险', '风险中'],
        '低风险':['风险低', '稳健类', '稳赚不赔', '稳健理财'],
        '高风险':['风险比较高', '风险高'],
        '中风险':['中等风险', '风险中'],
        '食品饮料':['食品类', '饮料类', '食品', '饮料'],
        '医药':['医疗类', '医疗的'],
        '医药生物':['医疗生物', '医疗类生物'],
        '小米概念':['小米'],
        '中高风险':['风险偏高'],
        '证券':['券商'],
        '短期理财':['时间短期'],
        '指数型':['指数基金'],
        '债券型':['债券基金', '债基'],
        '稳健':['稳建'],
        '券':['劵'],
        '保本型':['保本保息'],
    }

    if bantoken not in text:
        if bantoken in reverse_dict.keys():
            for val in reverse_dict[bantoken]:
                if val in text:
                    text = text.replace(val, '')
                    break
    else:
        text = text.replace(bantoken, '')

    return text


class InputFeature(object):
    def __init__(self,
                 question,
                 table_id,
                 tokens,
                 word_to_subword,
                 subword_to_word,
                 input_ids,
                 input_mask,
                 segment_ids):
        self.question = question
        self.table_id = table_id
        self.tokens = tokens
        self.word_to_subword = word_to_subword
        self.subword_to_word = subword_to_word
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.columns = None
        self.select = None
        self.where_num = None
        self.where = None
        self.op = None
        self.value_start = None
        self.value_end = None

    def output_SQ(self, sel = None, conditions = None, return_str=True):
        agg_ops = ['none', 'avg', 'max', 'min', 'count', 'sum']
        cond_ops = ['>', '<', '==', '!=', 'like', '>=', '<=']

        if sel is None and conditions is None:
            sel = np.argmax(self.select)
            conditions = []
            for i in range(len(self.where)):
                if self.where[i] == 0:
                    continue
                conditions.append((i, self.op[i], self.value_start[i], self.value_end[i]))

        select_text = self.columns[sel]
        cond_texts = []
        for wc, op, vs, ve in conditions:
            column_text = self.columns[wc]
            op_text = cond_ops[op]
            value_span_text = "".join(self.tokens[wc][vs:ve])
            value_span_text = value_span_text.replace('#', "")
            cond_texts.append(column_text + op_text + value_span_text.rstrip())

        if return_str:
            sq = select_text + ", " + " AND ".join(cond_texts)
        else:
            sq = (select_text, set(cond_texts))

        return sq

class HydraFeaturizer(object):
    def __init__(self, config):
        self.config = config
        self.tokenizer = utils.create_tokenizer(config)

    def get_input_feature(self, example: SQLExample, config):
        max_total_length = int(config["max_total_length"])

        input_feature = InputFeature(
            example.question,
            example.table_id,
            [],
            [],
            [],
            [],
            [],
            []
        )

        for column, col_type, _ in example.column_meta:
            # get query tokens
            tokens = []
            word_to_subword = []
            subword_to_word = []
            for i, query_token in enumerate(example.tokens):
                if self.config["base_class"] == "roberta":
                    sub_tokens = self.tokenizer.tokenize(query_token)
                else:
                    sub_tokens = self.tokenizer.tokenize(query_token)
                cur_pos = len(tokens)
                if len(sub_tokens) > 0:
                    word_to_subword += [(cur_pos, cur_pos + len(sub_tokens))]
                    tokens.extend(sub_tokens)
                    subword_to_word.extend([i] * len(sub_tokens))


            tokenize_result = self.tokenizer.encode_plus(
                col_type + " " + column,
                "".join(tokens),
                padding="max_length",
                max_length=max_total_length,
                truncation_strategy="longest_first",
                truncation=True,
            )

            input_ids = tokenize_result["input_ids"]
            input_mask = tokenize_result["attention_mask"]

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            column_token_length = 0
            if self.config["base_class"] == "roberta":
                for i, token_id in enumerate(input_ids):
                    if token_id == self.tokenizer.sep_token_id:
                        column_token_length = i + 2
                        break
                segment_ids = [0] * max_total_length
                for i in range(column_token_length, max_total_length):
                    if input_mask[i] == 0:
                        break
                    segment_ids[i] = 1
            else:
                for i, token_id in enumerate(input_ids):
                    if token_id == self.tokenizer.sep_token_id:
                        column_token_length = i + 1
                        break
                segment_ids = tokenize_result["token_type_ids"]

            subword_to_word = [0] * column_token_length + subword_to_word
            word_to_subword = [(pos[0]+column_token_length, pos[1]+column_token_length) for pos in word_to_subword]

            assert len(input_ids) == max_total_length
            assert len(input_mask) == max_total_length
            assert len(segment_ids) == max_total_length

            input_feature.tokens.append(tokens)
            input_feature.word_to_subword.append(word_to_subword)
            input_feature.subword_to_word.append(subword_to_word)
            input_feature.input_ids.append(input_ids)
            input_feature.input_mask.append(input_mask)
            input_feature.segment_ids.append(segment_ids)

        return input_feature

    def fill_label_feature(self, example: SQLExample, input_feature: InputFeature, config):
        max_total_length = int(config["max_total_length"])

        columns = [c[0] for c in example.column_meta]
        col_num = len(columns)
        input_feature.columns = columns

        input_feature.where_num = [len(example.conditions) if example.conditions else 0] * col_num

        input_feature.select = [0] * len(columns)
        input_feature.select[example.select] = 1

        input_feature.where = [0] * len(columns)
        input_feature.op = [0] * len(columns)
        input_feature.value_start = [0] * len(columns)
        input_feature.value_end = [0] * len(columns)

        if example.conditions:
            for colidx, op, _ in example.conditions:
                input_feature.where[colidx] = 1
                input_feature.op[colidx] = op
        for colidx, column_meta in enumerate(example.column_meta):
            if column_meta[-1] == None:
                continue

            se = list(example.value_start_end.values())

            try:
                for se_sample in se:
                    s = input_feature.word_to_subword[colidx][se_sample[0]][0]
                    input_feature.value_start[colidx] = s
                    e = input_feature.word_to_subword[colidx][se_sample[1]-1][1]    # -1
                    input_feature.value_end[colidx] = e
                    assert s < max_total_length and input_feature.input_mask[colidx][s] == 1
                    assert e < max_total_length and input_feature.input_mask[colidx][e] == 1

            except:
                print("value span is out of range")
                return False

        # feature_sq = input_feature.output_SQ(return_str=False)
        # example_sq = example.output_SQ(return_str=False)
        # if feature_sq != example_sq:
        #     print(example.qid, feature_sq, example_sq)
        return True

    def load_data(self, data_paths, config, include_label=False):

        model_inputs = {k: [] for k in ["input_ids", "input_mask", "segment_ids"]}
        if include_label:
            for k in ["select", "where_num", "where", "op", "value_start", "value_end"]:
                model_inputs[k] = []

        pos = []
        input_features = []
        for data_path in data_paths.split("|"):
            cnt = 0
            for line in open(data_path, encoding="utf8"):
                if include_label == False:
                    example = SQLExample.load_from_json_test(line)
                else:
                    example = SQLExample.load_from_json(line)
                if not example.valid and include_label == True:
                    continue

                input_feature = self.get_input_feature(example, config)
                if include_label:
                    success = self.fill_label_feature(example, input_feature, config)
                    if not success:
                        continue

                input_features.append(input_feature)

                cur_start = len(model_inputs["input_ids"])
                cur_sample_num = len(input_feature.input_ids)
                pos.append((cur_start, cur_start + cur_sample_num))

                model_inputs["input_ids"].extend(input_feature.input_ids)
                model_inputs["input_mask"].extend(input_feature.input_mask)
                model_inputs["segment_ids"].extend(input_feature.segment_ids)
                if include_label:
                    model_inputs["select"].extend(input_feature.select)
                    model_inputs["where_num"].extend(input_feature.where_num)
                    model_inputs["where"].extend(input_feature.where)
                    model_inputs["op"].extend(input_feature.op)
                    model_inputs["value_start"].extend(input_feature.value_start)
                    model_inputs["value_end"].extend(input_feature.value_end)

                cnt += 1
                if cnt % 5000 == 0:
                    print(cnt)

                if "DEBUG" in config and cnt > 100:
                    break

        for k in model_inputs:
            model_inputs[k] = np.array(model_inputs[k], dtype=np.int64)

        return input_features, model_inputs, pos

    # 二次解码数据处理
    def load_data_with_banlist(self, data_paths, config, banlist):

        model_inputs = {k: [] for k in ["input_ids", "input_mask", "segment_ids"]}
        pos = []
        input_features = []
        stopwords = ['和', '跟', '或者', '或', '以及']
        ban_index = list(banlist.keys())
        for data_path in data_paths.split("|"):
            cnt = 0
            for index, line in enumerate(open(data_path, encoding="utf8")):
                if index not in ban_index:
                    continue
                
                line_split = json.loads(line)
                line = line_split['question']
                # 反向打补丁，将之前处理过的数据还原以避免无法消减问题
                line = reverse_pad(line, banlist[index][1][0][-1])  
                # 去掉之前识别的内容
                for sw in stopwords:
                    line = line.replace(sw, '')
                
                line_split['question'] = line
                line = json.dumps(line_split)
                example = SQLExample.load_from_json_test(line)                
                input_feature = self.get_input_feature(example, config)
                input_features.append(input_feature)

                cur_start = len(model_inputs["input_ids"])
                cur_sample_num = len(input_feature.input_ids)
                pos.append((cur_start, cur_start + cur_sample_num))

                model_inputs["input_ids"].extend(input_feature.input_ids)
                model_inputs["input_mask"].extend(input_feature.input_mask)
                model_inputs["segment_ids"].extend(input_feature.segment_ids)

                cnt += 1
                if cnt % 5000 == 0:
                    print(cnt)

                if "DEBUG" in config and cnt > 100:
                    break

        for k in model_inputs:
            model_inputs[k] = np.array(model_inputs[k], dtype=np.int64)

        return input_features, model_inputs, pos
    
class SQLDataset(torch_data.Dataset):
    def __init__(self, data_paths, config, featurizer, include_label=False, banlist=None):
        self.config = config
        self.featurizer = featurizer
        if not banlist:
            self.input_features, self.model_inputs, self.pos = self.featurizer.load_data(data_paths, config, include_label)
        else:
            self.input_features, self.model_inputs, self.pos = self.featurizer.load_data_with_banlist(data_paths, config, banlist)

        print("{0} loaded. Data shapes:".format(data_paths))
        for k, v in self.model_inputs.items():
            print(k, v.shape)

    def __len__(self):
        return self.model_inputs["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.model_inputs.items()}
