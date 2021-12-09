import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from transformers import AutoModel, AutoTokenizer, AutoConfig

import matplotlib.pyplot as plt

plt.style.use('seaborn-talk')
from colorama import Fore, Back, Style
from sklearn.metrics import log_loss

import torch
import json
import sys

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
    arg("--settings", default="./nishika-narou-2021-solution/settings.json", type=str, help="settings path")
    arg("--is_test", action="store_true", help="test")
    return parser


args = make_parse().parse_args()

with open(args.settings) as f:
    js = json.load(f)


class Config:
    model_name = js["model_name"]
    output_hidden_states = js["output_hidden_states"]
    batch_size = js["batch_size"]
    device = js["device"]
    seed = js["seed"]
    train_dir = js["train_dir"]
    item = js["item"]
    dataset_dir = js["dataset_dir"]
    item_num = js["i41"]["item_num"]  # 0ならstory(あらすじ),1ならtitle(題名),2ならkeyword(タグ)
    output_dir = js["i41"]["output_dir"]
    max_len = js["i41"]["max_len"]
    model_dir = js["models_dir"] + "/" + js["i41"]["model_dir"]
    narou_dir = js["narou_dir"]


sys.path.append(Config.narou_dir)

from utils.preprocess import remove_url
from utils.model import NarouModel
from utils.dataset import NishikaNarouDataset

os.system('pip install transformers fugashi ipadic unidic_lite --quiet')
os.system('mkdir -p ' + Config.output_dir)

test_df = pd.read_csv(Config.dataset_dir + '/test.csv')
test_df.head()
if Config.item[Config.item_num] == 'keyword':
    test_df.keyword[test_df.keyword.isnull()] = 'None'
test_df['excerpt'] = test_df[Config.item[Config.item_num]]

test_df.excerpt = test_df.excerpt.replace('\n', '', regex=True)
test_df.excerpt = test_df.excerpt.map(remove_url)

if Config.item[Config.item_num] == 'keyword':
    test_df.excerpt = test_df.excerpt.replace(' ', 'で', regex=True)

tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
config = AutoConfig.from_pretrained(Config.model_name)
config.update({
    "hidden_dropout_prob": 0.0,
    "layer_norm_eps": 1e-7,
    "output_hidden_states": True
})
transformer = AutoModel.from_pretrained(Config.model_name, config=config)

models_preds = []
n_models = 5

for model_num in range(n_models):
    print(f'Inference#{model_num + 1}/{n_models}')
    test_ds = NishikaNarouDataset(data=test_df, tokenizer=tokenizer, Config=Config, is_test=True)
    test_sampler = SequentialSampler(test_ds)
    test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=Config.batch_size)

    model = NarouModel(transformer, config)
    model.load_state_dict(torch.load(Config.model_dir + f'/best_model_{model_num}.pt'))
    model = model.to(Config.device)

    all_preds = []
    model.eval()

    for step, batch in enumerate(test_dataloader):
        sent_id, mask = batch['input_ids'].to(Config.device), batch['attention_mask'].to(Config.device)
        with torch.no_grad():
            preds = model(sent_id, mask)
            all_preds += preds.cpu().tolist()

    models_preds.append(all_preds)

models_preds = np.array(models_preds)
m_preds = models_preds.mean(axis=0)
all_preds = softmax(np.reshape(m_preds, (-1, 5)), axis=1)
result_df = pd.DataFrame(
    {
        'ncode': test_df.ncode,
        'proba_0': all_preds[:, 0],
        'proba_1': all_preds[:, 1],
        'proba_2': all_preds[:, 2],
        'proba_3': all_preds[:, 3],
        'proba_4': all_preds[:, 4]

    })

result_df.to_csv(Config.output_dir + '/submission.csv', index=False)

if args.is_test:
    print("exit: test_mode")
    sys.exit(0)


def loss_fn(y_true, y_pred):
    return log_loss(y_true, softmax(y_pred, axis=1))


models_preds = []

test_df = pd.read_csv(Config.train_dir + '/kfold_from_2020_to_2021_06.csv')

if Config.item[Config.item_num] == 'keyword':
    test_df.keyword[test_df.keyword.isnull()] = 'None'
test_df['excerpt'] = test_df[Config.item[Config.item_num]]

test_df.excerpt = test_df.excerpt.replace('\n', '', regex=True)
test_df.excerpt = test_df.excerpt.map(remove_url)

if Config.item[Config.item_num] == 'keyword':
    test_df.excerpt = test_df.excerpt.replace(' ', 'で', regex=True)

test_df['target'] = test_df['fav_novel_cnt_bin']

score = []
models_preds = []
all_val_pre_df = pd.DataFrame()
for model_num in range(n_models):
    print(f'Inference#{model_num + 1}/{n_models}')
    test_ds = NishikaNarouDataset(data=test_df[test_df['fold'] == model_num], tokenizer=tokenizer, Config=Config,
                                  is_test=True)
    test_sampler = SequentialSampler(test_ds)
    test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=Config.batch_size)

    model = NarouModel(transformer, config)
    model.load_state_dict(torch.load(Config.model_dir + f'/best_model_{model_num}.pt'))
    model = model.to(Config.device)

    all_preds = []
    model.eval()

    for step, batch in enumerate(test_dataloader):
        sent_id, mask = batch['input_ids'].to(Config.device), batch['attention_mask'].to(Config.device)
        with torch.no_grad():
            preds = model(sent_id, mask)
            all_preds += preds.cpu().tolist()

    models_preds.append(all_preds)

    y_true = np.array(test_df['target'][test_df['fold'] == model_num])
    all_preds = np.array(all_preds)
    score.append(loss_fn(y_true, all_preds))

    val_pre_df = pd.concat(
        [test_df[test_df['fold'] == model_num].ncode.reset_index(drop=True), pd.DataFrame(all_preds)], axis=1)

    all_val_pre_df = pd.concat([all_val_pre_df, val_pre_df], axis=0)
print('cv')
print(np.mean(score))

all_val_pre_df.to_csv(Config.output_dir + '/val_pred.csv', index=False)

models_preds = []

test_df = pd.read_csv(Config.train_dir + '/kfold_2021_06.csv')

if Config.item[Config.item_num] == 'keyword':
    test_df.keyword[test_df.keyword.isnull()] = 'None'
test_df['excerpt'] = test_df[Config.item[Config.item_num]]

test_df.excerpt = test_df.excerpt.replace('\n', '', regex=True)
test_df.excerpt = test_df.excerpt.map(remove_url)

if Config.item[Config.item_num] == 'keyword':
    test_df.excerpt = test_df.excerpt.replace(' ', 'で', regex=True)

test_df['target'] = test_df['fav_novel_cnt_bin']

score = []
models_preds = []
all_val_pre_df = pd.DataFrame()
for model_num in range(n_models):
    print(f'Inference#{model_num + 1}/{n_models}')
    test_ds = NishikaNarouDataset(data=test_df, tokenizer=tokenizer, Config=Config, is_test=True)
    test_sampler = SequentialSampler(test_ds)
    test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=Config.batch_size)

    model = NarouModel(transformer, config)
    model.load_state_dict(torch.load(Config.model_dir + f'/best_model_{model_num}.pt'))
    model = model.to(Config.device)

    all_preds = []
    model.eval()

    for step, batch in enumerate(test_dataloader):
        sent_id, mask = batch['input_ids'].to(Config.device), batch['attention_mask'].to(Config.device)
        with torch.no_grad():
            preds = model(sent_id, mask)
            all_preds += preds.cpu().tolist()

    models_preds.append(all_preds)

    y_true = np.array(test_df['target'])
    all_preds = np.array(all_preds)
    score.append(loss_fn(y_true, all_preds))
print('cv')
print(np.mean(score))

val_pre_df = pd.concat([test_df.ncode.reset_index(drop=True), pd.DataFrame(list(np.array(models_preds).mean(axis=0)))],
                       axis=1)
val_pre_df.to_csv(Config.output_dir + '/kfold_2021_06_val_pred.csv', index=False)
