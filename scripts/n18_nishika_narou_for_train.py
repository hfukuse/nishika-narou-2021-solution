import os
import warnings
import sys
import json

import numpy as np
import pandas as pd
import argparse

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, AutoConfig
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

plt.style.use("seaborn-talk")
from colorama import Fore, Back, Style

r_ = Fore.RED
b_ = Fore.BLUE
g_ = Fore.GREEN
y_ = Fore.YELLOW
w_ = Fore.WHITE
bb_ = Back.BLACK
sr_ = Style.RESET_ALL



def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--debug",default=False ,action="store_true", help="debug")
    arg("--settings", default="./nishika-narou-2021-1st-place-solution/settings.json", type=str, help="settings path")
    arg("--is_test", action="store_true", help="test")
    return parser

args = make_parse().parse_args()

with open(args.settings) as f:
    js = json.load(f)

debug=args.debug

class Config:
    model_name = js["model_name"]
    pretrained_model_path = js["pretrained_model_path"]
    output_hidden_states = js["output_hidden_states"]
    batch_size = js["batch_size"]
    device = js["device"]
    seed = js["seed"]
    train_dir = js["train_dir"]
    item = js["item"]
    dataset_dir = js["dataset_dir"]
    item_num = js["i18"]["item_num"] # 0ならstory(あらすじ),1ならtitle(題名),2ならkeyword(タグ)
    output_dir = js["i18"]["output_dir"]
    max_len = js["i18"]["max_len"]
    model_dir = js["models_dir"]+"/"+js["i18"]["model_dir"]
    narou_dir = js["narou_dir"]
    epochs = js["epochs"]
    max_len = 256
    lr = 1e-5
    wd = 0.01
    eval_schedule = [(float("inf"), 40), (0.85, 30), (0.80, 20), (0.70, 10), (0, 0)]
    gradient_accumulation = 2
if debug:
    Config.epochs = 1
    Config.max_len = 5
    Config.eval_schedule = [(float("inf"), 500), (0, 500)]

sys.path.append(Config.narou_dir)

from utils.preprocess import remove_url
from utils.model import NarouModel
from utils.dataset import NishikaNarouDataset
from utils.Trainer import AvgCounter,EvaluationScheduler,DynamicPadCollate,Trainer
from utils.optimizer import create_optimizer
from utils.utils import seed_everything

def make_dataloader(data, tokenizer, is_train=True):
    dataset = NishikaNarouDataset(data, tokenizer=tokenizer, max_len=Config.max_len)
    if is_train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    batch_dataloader = DataLoader(dataset, sampler=sampler, batch_size=Config.batch_size, pin_memory=True,
                                  collate_fn=DynamicPadCollate())
    return batch_dataloader


def loss_fn(y_true, y_pred):
    return nn.functional.cross_entropy(y_true, y_pred.to(torch.long))

def main():
    os.system("pip install transformers fugashi ipadic unidic_lite --quiet")
    os.system("mkdir -p " + Config.models_dir)

    seed_everything(seed=Config.seed)

    kfold_df = pd.read_csv(Config.train_dir + "/kfold_2021_06.csv")

    kfold_df["excerpt"] = kfold_df[Config.item[Config.item_num]]
    kfold_df.excerpt = kfold_df.excerpt.replace("\n", "", regex=True)

    kfold_df.excerpt = kfold_df.excerpt.map(remove_url)

    kfold_df["target"] = kfold_df["fav_novel_cnt_bin"]

    best_scores = []
    for model_num in range(5):
        print(f"{bb_}{w_}  Model#{model_num + 1}  {sr_}")

        tokenizer = AutoTokenizer.from_pretrained(Config.pretrained_model_path)
        config = AutoConfig.from_pretrained(Config.pretrained_model_path)
        config.update({
            "hidden_dropout_prob": 0.0,
            "layer_norm_eps": 1e-7,
            "output_hidden_states": True
        })

        train_dl = make_dataloader(kfold_df[kfold_df.fold != model_num], tokenizer)
        val_dl = make_dataloader(kfold_df[kfold_df.fold == model_num], tokenizer, is_train=False)
        transformer = AutoModel.from_pretrained(Config.pretrained_model_path, config=config)

        model = NarouModel(transformer, config)

        model = model.to(Config.device)
        optimizer = create_optimizer(model, Config.lr)
        scaler = GradScaler()
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=Config.epochs * len(train_dl),
            num_warmup_steps=len(train_dl) * Config.epochs * 0.11)

        criterion = loss_fn

        trainer = Trainer(train_dl, val_dl, model, optimizer, scheduler, scaler, criterion, model_num, Config)
        record_info, best_val_loss = trainer.run()
        best_scores.append(best_val_loss)

        steps, train_losses = list(zip(*record_info["train_loss"]))
        plt.plot(steps, train_losses, label="train_loss")
        steps, val_losses = list(zip(*record_info["val_loss"]))
        plt.plot(steps, val_losses, label="val_loss")
        plt.legend()
        plt.show()

    print("Best val losses:", best_scores)
    print("Avg val loss:", np.array(best_scores).mean())

if __name__ == "__main__":
    main()
