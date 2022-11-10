import os
import json
from transformers import AutoTokenizer, AutoModel

pretrained_weights = {
    ("nezha", "base"): "sijunhe/nezha-base-wwm",
    ("roberta", "base"): "hfl/chinese-roberta-wwm-ext",
    ("finbert", "base"): "./FinBERT_L-12_H-768_A-12_pytorch",
    ("macbert", 'base'): "Langboat/mengzi-bert-base-fin"
}


def read_jsonl(jsonl):
    for line in open(jsonl, encoding="utf8"):
        sample = json.loads(line.rstrip())
        yield sample

def read_conf(conf_path):
    config = {}
    for line in open(conf_path, encoding="utf8"):
        if line.strip() == "" or line[0] == "#":
             continue
        fields = line.strip().split("\t")
        config[fields[0]] = fields[1]
    config["train_data_path"] =  os.path.abspath(config["train_data_path"])
    config["dev_data_path"] =  os.path.abspath(config["dev_data_path"])

    return config

def create_base_model(config):
    weights_name = pretrained_weights[(config["base_class"], config["base_name"])]
    return AutoModel.from_pretrained(weights_name)

def create_tokenizer(config):
    weights_name = pretrained_weights[(config["base_class"], config["base_name"])]
    return AutoTokenizer.from_pretrained(weights_name, add_prefix_space=True)


def preprocess_cond(text):
    tmp = text
    if '基金管理有限公司' in tmp:
        tmp = tmp.replace('基金管理有限公司', '')

    if '资产管理有限公司' in tmp:
        tmp = tmp.replace('资产管理有限公司', '')

    if '基金管理股份有限公司' in tmp:
        tmp = tmp.replace('基金管理股份有限公司', '')

    if '基金管理有限责任公司' in tmp:    
        tmp = tmp.replace('基金管理有限责任公司', '')

    if '股份有限公司' in tmp:    
        tmp = tmp.replace('股份有限公司', '')

    if '有限公司' in tmp:    
        tmp = tmp.replace('有限公司', '')

    return tmp

if __name__ == "__main__":
    qtokens = ['Tell', 'me', 'what', 'the', 'notes', 'are', 'for', 'South', 'Australia']
    column = "string School/Club Team"

    tokenizer = create_tokenizer({"base_class": "roberta", "base_name": "large"})

    qsubtokens = []
    for t in qtokens:
        qsubtokens += tokenizer.tokenize(t, add_prefix_space=True)
    print(qsubtokens)
    result = tokenizer.encode_plus(column, qsubtokens, add_prefix_space=True)
    for k in result:
        print(k, result[k])
    print(tokenizer.convert_ids_to_tokens(result["input_ids"]))



