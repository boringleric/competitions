import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import json
from torch_model import HydraTorch
import utils
from featurizer import HydraFeaturizer, SQLDataset
from post_process import post_process

if __name__ == "__main__":

    in_file = "./data/waic_nl2sql_testb_public.jsonl"
    out_file = "./output/20220824_232123.jsonl" 
    fin_file = './submit/20220824_232123.jsonl'
    model_path = "output/20220824_232123"
    epoch = 1

    config = utils.read_conf(os.path.join(model_path, "model.conf"))
    featurizer = HydraFeaturizer(config)
    pred_data = SQLDataset(in_file, config, featurizer, False)
    print("num of samples: {0}".format(len(pred_data.input_features)))

    model = HydraTorch(config)
    model.load(model_path, epoch)

    model_outputs = model.dataset_inference(pred_data)

    pred_sqls = model.predict_SQL(pred_data, model_outputs=model_outputs)
    back_ind = {}
    for index, pred in enumerate(pred_sqls):
        if pred[-1] == 2:
            back_ind[index] = pred

    # 针对cond=2进行二次推理
    pred_data2 = SQLDataset(in_file, config, featurizer, False, back_ind)
    model_outputs2 = model.dataset_inference(pred_data2)
    pred_sqls2 = model.predict_SQL(pred_data2, model_outputs=model_outputs2)
    outind = 0
    with open(out_file, "w") as g:
        for ind, pred_sql in enumerate(pred_sqls):
            if pred_sql[-1] == 2:
                result = {"query": {}}
                result["query"]["sel"] = int(pred_sql[0])
                a = (int(pred_sql[1][0][0]), int(pred_sql[1][0][1]), str(pred_sql[1][0][2]))
                oi = pred_sqls2[outind]
                if oi[-1] != 0:
                    b = (int(pred_sqls2[outind][1][0][0]), int(pred_sqls2[outind][1][0][1]), str(pred_sqls2[outind][1][0][2]))
                    result["query"]["conds"] = [a, b]
                else:
                    result["query"]["conds"] = [a]
                outind += 1
            else:
                result = {"query": {}}
                result["query"]["sel"] = int(pred_sql[0])
                result["query"]["conds"] = [(int(cond[0]), int(cond[1]), str(cond[2])) for cond in pred_sql[1]]
            g.write(json.dumps(result, ensure_ascii=False) + "\n")

    # 后处理
    post_process(out_file, in_file, fin_file)

    print('fin!')
