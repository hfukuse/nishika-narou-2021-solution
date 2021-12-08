import warnings
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import argparse
import json
import sys
sys.path.append("./nishika-narou-2021-1st-place-solution")
warnings.filterwarnings('ignore')

seed = 42

def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--settings", default="./nishika-narou-2021-1st-place-solution/settings_for_kaggle.json", type=str, help="settings path")
    return parser


args = make_parse().parse_args()

with open(args.settings) as f:
    js = json.load(f)

class Config:
    train_dir = js["train_dir"]
    dataset_dir = js["dataset_dir"]

os.system ('mkdir -p '+Config.train_dir)

# create kfold from 2021-06 to last
train_data = pd.read_csv(Config.dataset_dir+'/train.csv')
bins = train_data.fav_novel_cnt_bin
for i, data in enumerate(train_data.general_firstup):
    if data[0:7] == '2021-06':
        print(i)
        break
train_data = train_data.iloc[i:]
train_data = train_data.reset_index(drop=True)
bins = train_data.fav_novel_cnt_bin
train_data['fold'] = 0
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for fold, (train_idx, valid_idx) in enumerate(kfold.split(X=train_data, y=bins)):
    train_data.loc[valid_idx, 'fold'] = fold
train_data.to_csv(Config.train_dir+'/kfold_2021_06.csv', index=False)

# create kfold from 2020 to 2021-06
train_data = pd.read_csv(Config.dataset_dir+'/train.csv')
for i, data in enumerate(train_data.general_firstup):
    if data[0:4] == '2020':
        print(i)
        break
for j, data in enumerate(train_data.general_firstup):
    if data[0:7] == '2021-06':
        print(j)
        break
train_data = train_data.iloc[i:j]
train_data = train_data.reset_index(drop=True)
bins = train_data.fav_novel_cnt_bin
train_data['fold'] = 0
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for fold, (train_idx, valid_idx) in enumerate(kfold.split(X=train_data, y=bins)):
    train_data.loc[valid_idx, 'fold'] = fold
train_data.to_csv(Config.train_dir+'/kfold_from_2020_to_2021_06.csv', index=False)

# create kfold from begin to 2020
train_data = pd.read_csv(Config.dataset_dir+'/train.csv')
for j, data in enumerate(train_data.general_firstup):
    if data[0:4] == '2020':
        print(j)
        break
train_data = train_data.iloc[:j].reset_index(drop=True)
bins = train_data.fav_novel_cnt_bin
train_data['fold'] = 0
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for fold, (train_idx, valid_idx) in enumerate(kfold.split(X=train_data, y=bins)):
    train_data.loc[valid_idx, 'fold'] = fold
train_data.to_csv(Config.train_dir+'/kfold_from_begin_to_2020.csv', index=False)

# create kfold from 2020-07 to last
train_data = pd.read_csv(Config.dataset_dir+'/train.csv')
for i, data in enumerate(train_data.general_firstup):
    if data[0:7] == '2021-07':
        print(i)
        break
train_data = train_data.iloc[i:].reset_index(drop=True)
bins = train_data.fav_novel_cnt_bin
train_data['fold'] = 0
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for fold, (train_idx, valid_idx) in enumerate(kfold.split(X=train_data, y=bins)):
    train_data.loc[valid_idx, 'fold'] = fold
train_data.to_csv(Config.train_dir+'/kfold_2021_07.csv', index=False)

# create train_test_split
train_data = pd.read_csv(Config.dataset_dir+'/train.csv')
train_data = train_data.reset_index(drop=True)
bins = train_data.fav_novel_cnt_bin
X_train, X_test, y_train, y_test = train_test_split(train_data, bins, test_size=0.1, random_state=42, stratify=bins)
X_train['fold'] = 'train'
X_test['fold'] = 'valid'
train_data = pd.concat([X_train, X_test])
train_data.to_csv(Config.train_dir+'/train_stratify.csv', index=False)
