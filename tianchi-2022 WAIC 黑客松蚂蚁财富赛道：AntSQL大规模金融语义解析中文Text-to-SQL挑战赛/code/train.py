import argparse
import os
import shutil
import datetime
from torch_model import HydraTorch
import utils
from featurizer import HydraFeaturizer, SQLDataset
from evaluator import HydraEvaluator
import torch.utils.data as torch_data
import torch
import random
import numpy as np

def seed(seed=42):
    torch.manual_seed(seed)    
    random.seed(seed)    
    np.random.seed(seed)

seed()

parser = argparse.ArgumentParser(description='HydraNet training script')
parser.add_argument("--conf", default='./conf/wikisql.conf', help="conf file path")
parser.add_argument("--output_path", type=str, default="./output", help="folder path for all outputs")
parser.add_argument("--model_path", default="model", help="trained model folder path (used in eval, predict and export mode)")
parser.add_argument("--gpu", type=str, default='0', help="gpu id")

args = parser.parse_args()

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
conf_path = os.path.abspath(args.conf)
config = utils.read_conf(conf_path)

output_path = args.output_path
model_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(output_path, model_name)

if "DEBUG" not in config:
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    shutil.copyfile(conf_path, os.path.join(model_path, "model.conf"))

featurizer = HydraFeaturizer(config)
train_data = SQLDataset(config["train_data_path"], config, featurizer, True)
train_data_loader = torch_data.DataLoader(train_data, batch_size=int(config["batch_size"]))

num_samples = len(train_data)
config["num_train_steps"] = int(num_samples * int(config["epochs"]) / int(config["batch_size"]))
step_per_epoch = num_samples / int(config["batch_size"])
print("total_steps: {0}, warm_up_steps: {1}".format(config["num_train_steps"], config["num_warmup_steps"]))

model = HydraTorch(config)

evaluator = HydraEvaluator(model_path, config, featurizer, model)
print("start training")
loss_avg, step, epoch = 0.0, 0, 0
while True:
    for batch_id, batch in enumerate(train_data_loader):
        cur_loss = model.train_on_batch(batch)
        loss_avg = (loss_avg * step + cur_loss) / (step + 1)
        step += 1
        if batch_id % 100 == 0:
            currentDT = datetime.datetime.now()
            print("[{3}] epoch {0}, batch {1}, batch_loss={2:.4f}".format(epoch, batch_id, cur_loss, currentDT.strftime("%m-%d %H:%M:%S")))

        if batch_id % 5000 == 0 and batch_id > 1:
            evaluator.eval(epoch)
            
    model.save(model_path, epoch)
    evaluator.eval(epoch)
    epoch += 1
    if epoch >= int(config["epochs"]):
        break
