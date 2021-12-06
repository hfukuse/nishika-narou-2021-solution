import datetime
import os
import re

import numpy as np
import pandas as pd
import tqdm
import argparse


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--settings", default="./settings_for_kaggle.json", type=str, help="settings path")
    return parser

args = make_parse().parse_args()

with open(args.settings) as f:
    js = json.load(f)

class Config:
    train_dir = js["train_dir"]
    dataset_dir = js["dataset_dir"]
    pos_dir = js["pos_dir"]

train_df = pd.read_csv(Config.train_dir + '/kfold_2021_06.csv')
train2_df = pd.read_csv(Config.train_dir + '/kfold_from_2020_to_2021_06.csv')
test_df = pd.read_csv(Config.dataset_dir + '/test.csv')
sub_df = pd.read_csv(Config.dataset_dir + '/sample_submission.csv')

test_df["fold"] = 6

# 作れる
df1 = pd.read_csv(Config.pos_dir + "/kfold_2021_06_with_pos.csv")
df2 = pd.read_csv(Config.pos_dir + "/kfold_from_2020_to_2021_06_with_pos.csv")
raw_df = pd.concat([df1, df2]).reset_index(drop=True)
raw_df = raw_df.drop_duplicates().reset_index(drop=True)


def create_tag_feature(df):
    arr = []
    key_w = ["ざまあ"]
    for i, item in df.iterrows():
        for k in key_w:
            if k in str(item.keyword):
                arr += [1]
            else:
                arr += [0]
    arr = np.array(arr).reshape(-1, len(key_w))
    return pd.DataFrame(arr, columns=key_w), key_w


tags_df, key_w = create_tag_feature(raw_df)

raw_df["tenni_tennsei"] = raw_df.istenni + raw_df.istensei

raw_df["genre_each_count"] = 0
raw_df["biggenre_each_count"] = 0
for i in set(raw_df.userid):
    genre_count = len(set(raw_df[raw_df.userid == i].genre))
    biggenre_count = len(set(raw_df[raw_df.userid == i].biggenre))
    raw_df.loc[raw_df.userid == i, "genre_each_count"] = genre_count
    raw_df.loc[raw_df.userid == i, "biggenre_each_count"] = biggenre_count

raw_df["past_days_from_previous_work"] = 0
for j in set(raw_df.userid):
    past_days_from_previous_work = [10000]
    past_days = raw_df[raw_df.userid == j].sort_values(by="general_firstup").past_days
    for i in range(len(past_days) - 1):
        past_days_from_previous_work.append(past_days.iloc[i] - past_days.iloc[i + 1])
    raw_df["past_days_from_previous_work"].loc[past_days.index] = past_days_from_previous_work

raw_df["past_days_from_previous_work_tyohen"] = 10000
raw_df["past_days_from_previous_work_tanpen"] = 10000
for j in set(raw_df.userid):
    past_days_from_previous_work_tyohen = [10000]
    past_days_from_previous_work_tanpen = [10000]

    tyohen_df = raw_df[raw_df.userid == j][(raw_df[raw_df.userid == j].novel_type == 1)].sort_values(
        by="general_firstup")
    tanpen_df = raw_df[raw_df.userid == j][(raw_df[raw_df.userid == j].novel_type == 2)].sort_values(
        by="general_firstup")

    past_days_tyohen = tyohen_df[tyohen_df.userid == j].past_days
    past_days_tanpen = tanpen_df[tanpen_df.userid == j].past_days

    for i in range(len(past_days_tyohen) - 1):
        past_days_from_previous_work_tyohen.append(past_days_tyohen.iloc[i] - past_days_tyohen.iloc[i + 1])
    for i in range(len(past_days_tanpen) - 1):
        past_days_from_previous_work_tanpen.append(past_days_tanpen.iloc[i] - past_days_tanpen.iloc[i + 1])
    if len(tyohen_df) > 0:
        raw_df["past_days_from_previous_work_tyohen"].loc[tyohen_df.index] = past_days_from_previous_work_tyohen
    if len(tanpen_df) > 0:
        raw_df["past_days_from_previous_work_tanpen"].loc[tanpen_df.index] = past_days_from_previous_work_tanpen

raw_df["tyohen_each_count"] = 0
raw_df["tanpen_each_count"] = 0
for i in set(raw_df.userid):
    tyohen_count = sum(raw_df[raw_df.userid == i].novel_type == 1)
    tanpen_count = sum(raw_df[raw_df.userid == i].novel_type == 2)
    raw_df.loc[raw_df.userid == i, "tyohen_each_count"] = tyohen_count
    raw_df.loc[raw_df.userid == i, "tanpen_each_count"] = tanpen_count

# 作れる
df = pd.read_csv(Config.train_dir + "/kfold_from_begin_to_2020.csv")
d = {}
sum_d = {}
for i in set(df.userid):
    d[i] = np.mean(df.loc[df["userid"] == i, :].fav_novel_cnt_bin)
    sum_d[i] = np.sum(df.loc[df["userid"] == i, :].fav_novel_cnt_bin)

raw_df["mean_fav"] = 0.0
raw_df["sum_fav"] = 0.0
for i in set(raw_df.userid):
    if i in d:
        raw_df.loc[raw_df["userid"] == i, ["mean_fav"]] = d[i]
    if i in sum_d:
        raw_df.loc[raw_df["userid"] == i, ["sum_fav"]] = sum_d[i]
raw_df["mean_fav"] = raw_df["mean_fav"].fillna(0)
raw_df["sum_fav"] = raw_df["sum_fav"].fillna(0)

concat_df = pd.concat([raw_df, tags_df], axis=1)

concat_df.to_csv("concat_df.csv", index=False)

