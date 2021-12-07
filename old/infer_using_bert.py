import os
import warnings
import argparse
import sys

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SequentialSampler
from transformers import AutoTokenizer

import matplotlib.pyplot as plt

plt.style.use("seaborn-talk")
from colorama import Fore, Back, Style
from sklearn.metrics import log_loss

import torch
import torch.nn as nn
import re

from scipy.special import softmax

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
    arg("--debug", action="store_true", help="debug")
    arg("--config", default="./configs/sample.yaml", type=str, help="config path")
    arg("--is_test", action="store_true", help="test")
    return parser

args = make_parse().parse_args()

class Config:
    model_name = "cl-tohoku/bert-base-japanese-v2"
    pretrained_model_path = "cl-tohoku/bert-base-japanese-v2"
    output_hidden_states = True
    epochs = 3
    batch_size = 16
    device = "cuda"
    seed = 42
    max_len = 256
    train_dir = "./train_val_split"
    item = ["story", "title", "keyword"]
    item_num = 2  # 0ならstory(あらすじ),1ならtitle(題名),2ならkeyword(タグ)
    output_dir = "i18_inference"
    dataset_dir = "../input/nishika-narou"
    model_dir = "../input/n8-nishika-narou-finetune-base-2019/models"
    loss_fn = "logloss"  # "rmse"


os.system("pip install transformers fugashi ipadic unidic_lite --quiet")
os.system("mkdir -p " + Config.output_dir)


class AttentionHead(nn.Module):
    def __init__(self, h_size, hidden_dim=512):
        super().__init__()
        self.W = nn.Linear(h_size, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector


class NarouModel(nn.Module):
    def __init__(self, transformer, config):
        super(NarouModel, self).__init__()
        self.h_size = config.hidden_size
        self.transformer = transformer
        self.head = AttentionHead(self.h_size * 4)
        self.linear = nn.Linear(self.h_size * 2, 5)
        self.linear_out = nn.Linear(self.h_size * 8, 5)

    def forward(self, input_ids, attention_mask):
        transformer_out = self.transformer(input_ids, attention_mask)

        all_hidden_states = torch.stack(transformer_out.hidden_states)
        cat_over_last_layers = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1
        )

        cls_pooling = cat_over_last_layers[:, 0]
        head_logits = self.head(cat_over_last_layers)
        y_hat = self.linear_out(torch.cat([head_logits, cls_pooling], -1))

        return y_hat


def convert_examples_to_features(text, tokenizer):
    tok = tokenizer.encode_plus(
        text,
        max_length=Config.max_len,
        truncation=True,
        padding="max_length")
    return tok


def remove_url(sentence):
    ret = re.sub(r"(http?|ftp)(:\/\/[-_\.!~*\"()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "", sentence)
    return ret



class NishikaNarouDataset(Dataset):
    def __init__(self, data, tokenizer, is_test=False):
        self.data = data
        self.excerpts = self.data.excerpt.tolist()
        if not is_test:
            self.targets = self.data.target.tolist()
        self.tokenizer = tokenizer
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if not self.is_test:
            excerpt = self.excerpts[item]
            label = self.targets[item]
            features = convert_examples_to_features(
                excerpt, self.tokenizer
            )
            return {
                "input_ids": torch.tensor(features["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(features["attention_mask"], dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.float),
            }
        else:
            excerpt = self.excerpts[item]
            features = convert_examples_to_features(
                excerpt, self.tokenizer
            )
            return {
                "input_ids": torch.tensor(features["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(features["attention_mask"], dtype=torch.long),
            }


test_df = pd.read_csv(Config.dataset_dir + "/test.csv")
test_df.head()
if Config.item[Config.item_num] == "keyword":
    test_df.keyword[test_df.keyword.isnull()] = "None"
test_df["excerpt"] = test_df[Config.item[Config.item_num]]

test_df.excerpt = test_df.excerpt.replace("\n", "", regex=True)
test_df.excerpt = test_df.excerpt.map(remove_url)

if Config.item[Config.item_num] == "keyword":
    test_df.excerpt = test_df.excerpt.replace(" ", "で",regex=True)



tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

models_preds = []
n_models = 5

for model_num in range(n_models):
    print(f"Inference#{model_num + 1}/{n_models}")
    test_ds = NishikaNarouDataset(data=test_df, tokenizer=tokenizer, is_test=True)
    test_sampler = SequentialSampler(test_ds)
    test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=Config.batch_size)

    model = torch.load(Config.model_dir + f"/best_model_{model_num}.pt").to(Config.device)

    all_preds = []
    model.eval()

    for step, batch in enumerate(test_dataloader):
        sent_id, mask = batch["input_ids"].to(Config.device), batch["attention_mask"].to(Config.device)
        with torch.no_grad():
            preds = model(sent_id, mask)
            all_preds += preds.cpu().tolist()

    models_preds.append(all_preds)

models_preds = np.array(models_preds)
m_preds = models_preds.mean(axis=0)
all_preds = softmax(np.reshape(m_preds, (-1, 5)), axis=1)
result_df = pd.DataFrame(
    {
        "ncode": test_df.ncode,
        "proba_0": all_preds[:, 0],
        "proba_1": all_preds[:, 1],
        "proba_2": all_preds[:, 2],
        "proba_3": all_preds[:, 3],
        "proba_4": all_preds[:, 4]

    })

result_df.to_csv(Config.output_dir + "/submission.csv", index=False)

if args.is_test:
    print("exit: test_mode")
    sys.exit(0)

def loss_fn(y_true, y_pred):
    return log_loss(y_true, softmax(y_pred, axis=1))


models_preds = []

test_df = pd.read_csv(Config.train_dir + "/kfold_2021_06.csv")

if Config.item[Config.item_num] == "keyword":
    test_df.keyword[test_df.keyword.isnull()] = "None"
test_df["excerpt"] = test_df[Config.item[Config.item_num]]

test_df.excerpt = test_df.excerpt.replace("\n", "", regex=True)
test_df.excerpt = test_df.excerpt.map(remove_url)

if Config.item[Config.item_num] == "keyword":
    test_df.excerpt = test_df.excerpt.replace(" ", "で",regex=True)

test_df["target"] = test_df["fav_novel_cnt_bin"]

score = []
models_preds = []
all_val_pre_df = pd.DataFrame()
for model_num in range(n_models):
    print(f"Inference#{model_num + 1}/{n_models}")
    test_ds = NishikaNarouDataset(data=test_df[test_df["fold"] == model_num], tokenizer=tokenizer, is_test=True)
    test_sampler = SequentialSampler(test_ds)
    test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=Config.batch_size)

    model = torch.load(Config.model_dir + f"/best_model_{model_num}.pt").to(
        Config.device)

    all_preds = []
    model.eval()

    for step, batch in enumerate(test_dataloader):
        sent_id, mask = batch["input_ids"].to(Config.device), batch["attention_mask"].to(Config.device)
        with torch.no_grad():
            preds = model(sent_id, mask)
            all_preds += preds.cpu().tolist()

    models_preds.append(all_preds)

    y_true = np.array(test_df["target"][test_df["fold"] == model_num])
    all_preds = np.array(all_preds)
    score.append(loss_fn(y_true, all_preds))
    print(score[-1])

    val_pre_df = pd.concat(
        [test_df[test_df["fold"] == model_num].ncode.reset_index(drop=True), pd.DataFrame(all_preds)], axis=1)

    all_val_pre_df = pd.concat([all_val_pre_df, val_pre_df], axis=0)
print("cv")
print(np.mean(score))

all_val_pre_df.to_csv(Config.output_dir + "/val_pred.csv", index=False)

models_preds = []

test_df = pd.read_csv(Config.train_dir + "/kfold_from_2020_to_2021_06.csv")

if Config.item[Config.item_num] == "keyword":
    test_df.keyword[test_df.keyword.isnull()] = "None"
test_df["excerpt"] = test_df[Config.item[Config.item_num]]

test_df.excerpt = test_df.excerpt.replace("\n", "", regex=True)
test_df.excerpt = test_df.excerpt.map(remove_url)

if Config.item[Config.item_num] == "keyword":
    test_df.excerpt = test_df.excerpt.replace(" ", "で",regex=True)

test_df["target"] = test_df["fav_novel_cnt_bin"]

score = []
models_preds = []
all_val_pre_df = pd.DataFrame()
for model_num in range(n_models):
    print(f"Inference#{model_num + 1}/{n_models}")
    test_ds = NishikaNarouDataset(data=test_df, tokenizer=tokenizer, is_test=True)
    test_sampler = SequentialSampler(test_ds)
    test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=Config.batch_size)

    model = torch.load(Config.model_dir + f"/best_model_{model_num}.pt").to(Config.device)

    all_preds = []
    model.eval()

    for step, batch in enumerate(test_dataloader):
        sent_id, mask = batch["input_ids"].to(Config.device), batch["attention_mask"].to(Config.device)
        with torch.no_grad():
            preds = model(sent_id, mask)
            all_preds += preds.cpu().tolist()

    models_preds.append(all_preds)

    y_true = np.array(test_df["target"])
    all_preds = np.array(all_preds)
    score.append(loss_fn(y_true, all_preds))
    print(Config.loss_fn+"_score")
    print(score[-1])
print("cv")
print(np.mean(score))

val_pre_df = pd.concat([test_df.ncode.reset_index(drop=True), pd.DataFrame(list(np.array(models_preds).mean(axis=0)))],
                       axis=1)
val_pre_df.to_csv(Config.output_dir + "/kfold_from_2020_to_2021_06_val_pred.csv", index=False)
