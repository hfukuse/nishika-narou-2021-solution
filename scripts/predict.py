import os
import sys

import json
import pickle
import re
from glob import glob
from tqdm import tqdm

import regex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import log_loss
import collections

from scipy.special import softmax
import catboost as cb

import seaborn as sns


class Config:
    model_name = 'cl-tohoku/bert-base-japanese-v2'
    pretrained_model_path = 'cl-tohoku/bert-base-japanese-v2'
    output_hidden_states = True
    device = 'cuda'
    seed = 42
    item = ['story', 'title', 'keyword']
    item_num = 0  # 0ならstory(あらすじ),1ならtitle(題名),2ならkeyword(タグ)
    output_dir = '.'
    train_dir = './train_val_split'
    dataset_dir = '../input/nishika-narou'
    model_dir = "../input/all-model-nishika-narou"
    pos_dir = "./nishika-narou-train-with-pos"

#作れる#with posはtrain_val_split dirへの統合はありかも
train_df = pd.read_csv(Config.train_dir+'/kfold_2021_06.csv')
train2_df = pd.read_csv(Config.train_dir+'/kfold_from_2020_to_2021_06.csv')
test_df = pd.read_csv(Config.dataset_dir+'/test.csv')
sub_df = pd.read_csv(Config.dataset_dir+'/sample_submission.csv')

test_df["fold"]=6


