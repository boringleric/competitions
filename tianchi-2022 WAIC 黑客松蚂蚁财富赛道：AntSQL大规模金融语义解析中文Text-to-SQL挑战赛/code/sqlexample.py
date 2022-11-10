import json
from transformers import AutoTokenizer

from query_dict import preprocessques, get_meta


tokenizer = AutoTokenizer.from_pretrained("./FinBERT_L-12_H-768_A-12_pytorch")
cnt_er = 0

class SQLExample(object):

    def __init__(self,
                 qid,
                 question,
                 table_id,
                 column_meta,
                 select=None,
                 conditions=None,
                 tokens=None,
                 value_start_end=None,
                 valid=True):
        self.qid = qid
        self.question = question
        self.table_id = table_id
        self.column_meta = column_meta
        self.select = select
        self.conditions = conditions
        self.valid = valid
        if tokens is None:
            self.tokens = tokenizer.tokenize(question)
            self.value_start_end = {}
            if conditions is not None and len(conditions) > 0:
                cur_start = None
                for cond in conditions:
                    value = cond[-1]
                    value_tokens = tokenizer.tokenize(value)
                    val_len = len(value_tokens)
                    for i in range(len(self.tokens)):
                        if " ".join(self.tokens[i:i+val_len]).lower() != " ".join(value_tokens).lower():
                            continue
                        s = i
                        e = len(value) + s
                        recovered_answer_text = question[s:e].strip()
                        if value.lower() == recovered_answer_text.lower():
                            cur_start = i
                            break

                    if cur_start is None:
                        global cnt_er
                        cnt_er += 1
                        self.valid = False
                    else:
                        self.value_start_end[value] = (cur_start, cur_start + val_len)
        else:
            self.tokens=tokens
            self.value_start_end=value_start_end

    @staticmethod
    def load_from_json(s):
        d = json.loads(s)
        keys = ["qid", "question", "table_id", "column_meta", "select", "conditions", "tokens", "value_start_end", "valid"]

        return SQLExample(*[d[k] for k in keys])

    @staticmethod
    def load_from_json_test(s):
        d = json.loads(s)
        keys = ["id", "question", "table_id"]

        return SQLExample(qid=d['id'], question=preprocessques(d['question'], False), table_id=d['table_id'], column_meta=get_meta())

    def dump_to_json(self):
        d = {}
        d["qid"] = self.qid
        d["question"] = self.question
        d["table_id"] = self.table_id
        d["column_meta"] = self.column_meta
        d["select"] = self.select
        d["conditions"] = self.conditions
        d["tokens"] = self.tokens
        d["value_start_end"] = self.value_start_end
        d["valid"] = self.valid

        return json.dumps(d, ensure_ascii=False)

    def output_SQ(self, return_str=True):
        agg_ops = ['none', 'avg', 'max', 'min', 'count', 'sum']
        cond_ops = ['>', '<', '==', '!=', 'like', '>=', '<=']

        select_text = self.column_meta[self.select][0]
        cond_texts = []
        for wc, op, value_text in self.conditions:
            column_text = self.column_meta[wc][0]
            op_text = cond_ops[op]
            cond_texts.append(column_text + op_text + value_text)

        if return_str:
            sq =  select_text + ", " + " AND ".join(cond_texts)
        else:
            sq = (select_text, set(cond_texts))
        return sq
