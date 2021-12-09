import argparse
import datetime
import json
import os
import re

import numpy as np
import pandas as pd

os.system('pip install xfeat --quiet')

from xfeat import Pipeline, SelectCategorical, LabelEncoder


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--settings", default="./nishika-narou-2021-solution/settings.json", type=str, help="settings path")
    return parser


args = make_parse().parse_args()

with open(args.settings) as f:
    js = json.load(f)


class Config:
    train_dir = js["train_dir"]
    dataset_dir = js["dataset_dir"]
    pos_dir = js["pos_dir"]


os.system('mkdir -p ' + Config.pos_dir)

train_df = pd.read_csv(Config.train_dir + '/kfold_2021_06.csv')
train2_df = pd.read_csv(Config.train_dir + '/kfold_from_2020_to_2021_06.csv')
test_df = pd.read_csv(Config.dataset_dir + '/test.csv')
sub_df = pd.read_csv(Config.dataset_dir + '/sample_submission.csv')

test_df["fold"] = 6

train_ind = len(train_df)
train2_ind = len(train_df) + len(test_df)

raw_df = pd.concat([train_df, test_df, train2_df])

raw_df["excerpt"] = raw_df["story"]
raw_df.story = raw_df.story.replace('\n', '', regex=True)


def remove_url(sen):
    ret = re.sub(r"(http?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "", sen)
    return ret


raw_df.story = raw_df.story.map(remove_url)

dt_now = datetime.datetime(2021, 9, 29, 0, 0, 0, 0)

train_idx = train_df.shape[0]
print(raw_df.shape)
raw_df.head(2)
TARGET = 'fav_novel_cnt_bin'

raw_df['past_days'] = raw_df['general_firstup'].apply(
    lambda x: (dt_now - datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)

encoder = Pipeline([
    SelectCategorical(),
    LabelEncoder(output_suffix=""),
])

le_df = encoder.fit_transform(raw_df)
le_df.head(2)

raw_df['writer'] = le_df['writer']
raw_df = raw_df.iloc[:train2_ind].reset_index(drop=True)

print(len(raw_df))


def create_type_features(texts):
    type_data = []

    for text in texts:
        tmp = []

        tmp.append(len(text))
        type_data.append(tmp)

    colnames = ['length']
    type_df = pd.DataFrame(type_data, columns=colnames)

    for colname in type_df.columns:
        if colname != 'length':
            type_df[colname] /= type_df['length']

    return type_df


titles = raw_df['title'].values
title_type_df = create_type_features(titles)
title_type_df.columns = ['title_' + colname for colname in title_type_df.columns]

stories = raw_df['story'].values

nrow_one_loop = 20000
nloop = np.floor(len(stories) / nrow_one_loop)
min_idx = 0

nrow_one_loop = 20000
nloop = np.floor(len(stories) / nrow_one_loop)
min_idx = 0

story_type_dfs = []

while min_idx < len(stories):
    tmp_stories = stories[min_idx:min_idx + nrow_one_loop]
    story_type_dfs.append(create_type_features(tmp_stories))
    min_idx += nrow_one_loop

story_type_df = pd.concat(story_type_dfs)
del story_type_dfs

raw_df = raw_df.reset_index(drop=True)

title_type_df = title_type_df.reset_index(drop=True)

story_type_df = story_type_df.reset_index(drop=True)

concat_df = pd.concat([raw_df, title_type_df, story_type_df], axis=1)

print(len(concat_df))


def convert_examples_to_features(text, tokenizer, max_len):
    tok = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=False
    )
    tok = tokenizer.decode(tok["input_ids"])
    return tok


concat_df.to_csv(Config.pos_dir + '/kfold_2021_06_with_pos.csv', index=False)

train_df = pd.read_csv(Config.train_dir + '/kfold_2021_06.csv')
train2_df = pd.read_csv(Config.train_dir + '/kfold_from_2020_to_2021_06.csv')
test_df = pd.read_csv(Config.dataset_dir + '/test.csv')
sub_df = pd.read_csv(Config.dataset_dir + '/sample_submission.csv')
test_df["fold"] = 6
train_ind = len(train_df)
train2_ind = len(train_df) + len(test_df)

raw_df = pd.concat([train_df, test_df, train2_df])

raw_df["excerpt"] = raw_df["story"]
raw_df.story = raw_df.story.replace('\n', '', regex=True)

raw_df.story = raw_df.story.map(remove_url)

train_idx = train_df.shape[0]
print(raw_df.shape)
raw_df.head(2)
TARGET = 'fav_novel_cnt_bin'

raw_df['past_days'] = raw_df['general_firstup'].apply(
    lambda x: (dt_now - datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)

encoder = Pipeline([
    SelectCategorical(),
    LabelEncoder(output_suffix=""),
])

le_df = encoder.fit_transform(raw_df)
le_df.head(2)

raw_df['writer'] = le_df['writer']
raw_df = raw_df.iloc[train_ind:].sort_values(by="general_firstup").reset_index(drop=True)

print(len(raw_df))


def create_type_features(texts):
    type_data = []
    for text in texts:
        tmp = []
        tmp.append(len(text))
        type_data.append(tmp)
    colnames = ['length']
    type_df = pd.DataFrame(type_data, columns=colnames)
    for colname in type_df.columns:
        if colname != 'length':
            type_df[colname] /= type_df['length']

    return type_df


titles = raw_df['title'].values

title_type_df = create_type_features(titles)
title_type_df.columns = ['title_' + colname for colname in title_type_df.columns]

stories = raw_df['story'].values

nrow_one_loop = 20000
nloop = np.floor(len(stories) / nrow_one_loop)
min_idx = 0

nrow_one_loop = 20000
nloop = np.floor(len(stories) / nrow_one_loop)
min_idx = 0

story_type_dfs = []

while min_idx < len(stories):
    tmp_stories = stories[min_idx:min_idx + nrow_one_loop]
    story_type_dfs.append(create_type_features(tmp_stories))
    min_idx += nrow_one_loop

story_type_df = pd.concat(story_type_dfs)
del story_type_dfs

raw_df.reset_index(drop=True, inplace=True)

title_type_df.reset_index(drop=True, inplace=True)

story_type_df.reset_index(drop=True, inplace=True)

concat_df = pd.concat([raw_df, title_type_df, story_type_df], axis=1)

concat_df.to_csv(Config.pos_dir + '/kfold_from_2020_to_2021_06_with_pos.csv', index=False)
