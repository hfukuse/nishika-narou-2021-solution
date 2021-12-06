import os

import catboost as cb
import numpy as np
import pandas as pd
from scipy.stats import logistic
from scipy.special import softmax
import re
import json

from utils.preprocess import remove_url,processing_ncode,count_keyword,count_nn_story,count_n_story

def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--debug", action="store_true", help="debug")
    arg("--settings", default="./nishika-narou-2021-1st-place-solution/settings_for_kaggle.json", type=str, help="settings path")
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
    pos_dir = js["pos_dir"]
    output_dir = js["n95"]["output_dir"]
    model_dir = js["models_dir"]+"/"+js["n95"]["model_dir"]


os.system('pip install transformers fugashi ipadic unidic_lite --quiet')
os.system('mkdir -p ' + Config.output_dir)

train_df = pd.read_csv(Config.train_dir + '/kfold_2021_06.csv')
train2_df = pd.read_csv(Config.train_dir + '/kfold_from_2020_to_2021_06.csv')
test_df = pd.read_csv(Config.dataset_dir + '/test.csv')
sub_df = pd.read_csv(Config.dataset_dir + '/sample_submission.csv')

test_df["fold"] = 6

raw_df = pd.concat([train_df, test_df])

raw_df["excerpt"] = raw_df["story"]
raw_df.story = raw_df.story.replace('\n', '', regex=True)

raw_df.story = raw_df.story.map(remove_url)

df1 = pd.read_csv(Config.pos_dir + "/kfold_2021_06_with_pos.csv")
df2 = pd.read_csv(Config.pos_dir + "/kfold_from_2020_to_2021_06_with_pos.csv")
raw_df = pd.concat([df1, df2]).reset_index(drop=True)

raw_df = raw_df.drop_duplicates().reset_index(drop=True)


def create_tag_feature(df):
    arr = []
    key_w = ["ざまあ", "ざまぁ", "追放", "TS", "女主人公", "チート", "婚約破棄", "令嬢", "ハーレム"]
    for i, item in df.iterrows():
        for k in key_w:
            if k in str(item.keyword):
                arr += [1]
            else:
                arr += [0]
    arr = np.array(arr).reshape(-1, len(key_w))
    return pd.DataFrame(arr, columns=key_w), key_w


tags_df, key_w = create_tag_feature(raw_df)


def create_shosekika_feature(df):
    arr = []
    key_w = ["書籍化", "日間", "年間", "コミカライズ"]
    for i, item in df.iterrows():
        for k in key_w:
            if k in str(item.story):
                arr += [1]
            else:
                arr += [0]
    arr = np.array(arr).reshape(-1, len(key_w))
    return pd.DataFrame(arr, columns=key_w), key_w


shosekika_df, story_w = create_shosekika_feature(raw_df)


def create_story_feature(df):
    arr = []
    key_w = ["ざまあ", "ざまぁ", "追放", "チート", "婚約破棄", "令嬢", "ハーレム"]
    for i, item in df.iterrows():
        for k in key_w:
            if k in str(item.story):
                arr += [1]
            else:
                arr += [0]
    key_w=["s_ざまあ", "s_ざまぁ", "s_追放", "s_チート", "s_婚約破棄", "s_令嬢", "s_ハーレム"]
    arr = np.array(arr).reshape(-1, len(key_w))
    return pd.DataFrame(arr, columns=key_w), key_w


sto_df, sto_w = create_story_feature(raw_df)


def create_title_feature(df):
    arr = []
    key_w = ["~", "〜", "【"]
    for i, item in df.iterrows():
        for k in key_w:
            if k in str(item.title):
                arr += [1]
            else:
                arr += [0]
    arr = np.array(arr).reshape(-1, len(key_w))
    return pd.DataFrame(arr, columns=["~", "〜", "【"]), ["~", "〜", "【"]


title_df, title_w = create_title_feature(raw_df)

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
    past_days = raw_df[raw_df.userid == j].past_days
    for i in range(len(past_days) - 1):
        past_days_from_previous_work.append(past_days.iloc[i] - past_days.iloc[i + 1])
    raw_df.loc[raw_df.userid == j, "past_days_from_previous_work"] = past_days_from_previous_work

raw_df["tanpen_each_count"] = 0
raw_df["tyohen_each_count"] = 0
for i in set(raw_df.userid):
    tyohen_count = sum(raw_df[raw_df.userid == i].novel_type == 1)
    tanpen_count = sum(raw_df[raw_df.userid == i].novel_type == 2)
    raw_df.loc[raw_df.userid == i, "tyohen_each_count"] = tyohen_count
    raw_df.loc[raw_df.userid == i, "tanpen_each_count"] = tanpen_count

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

concat_df = pd.concat([raw_df, tags_df, shosekika_df, sto_df, title_df], axis=1)
concat_df.shape

cat_cols = ['writer', 'biggenre', 'genre', 'novel_type', 'isr15', 'isbl', 'isgl', 'iszankoku', "tenni_tennsei",
            'pc_or_k'] + key_w + story_w + sto_w + title_w
num_cols = ['userid', 'past_days', 'title_length', 'length']
num_cols += ["past_days_from_previous_work"]
num_cols += ["genre_each_count", "biggenre_each_count"]
num_cols += ["mean_fav", "sum_fav"]
num_cols += ["tanpen_each_count", "tyohen_each_count"]

feat_cols = cat_cols + num_cols

ID = 'ncode'
TARGET = 'fav_novel_cnt_bin'

te_pred = pd.read_csv("i9_inference/submission.csv")
val_pred = pd.read_csv("i9_inference/val_pred.csv")
val2_pred = pd.read_csv("i9_inference/kfold_from_2020_to_2021_06_val_pred.csv")
val_pred.columns = te_pred.columns
val2_pred.columns = te_pred.columns
val_pred.iloc[:, 1:] = softmax(np.array(val_pred.iloc[:, 1:]), axis=1)
val2_pred.iloc[:, 1:] = softmax(np.array(val2_pred.iloc[:, 1:]), axis=1)
bert_df = pd.concat([te_pred, val_pred, val2_pred], axis=0).reset_index(drop=True)

num_cols += list(bert_df.columns)[1:]
feat_cols = cat_cols + num_cols

concat_df = pd.merge(concat_df, bert_df)

te_pred = pd.read_csv("i8_inference/submission.csv")
val_pred = pd.read_csv("i8_inference/val_pred.csv")
val2_pred = pd.read_csv("i8_inference/kfold_from_2020_to_2021_06_val_pred.csv")

te_pred.columns = ["ncode", "t_proba_0", "t_proba_1", "t_proba_2", "t_proba_3", "t_proba_4"]
val_pred.columns = te_pred.columns
val2_pred.columns = te_pred.columns
val_pred.iloc[:, 1:] = softmax(np.array(val_pred.iloc[:, 1:]), axis=1)
val2_pred.iloc[:, 1:] = softmax(np.array(val2_pred.iloc[:, 1:]), axis=1)
bert_df = pd.concat([te_pred, val_pred, val2_pred], axis=0).reset_index(drop=True)

num_cols += ["t_proba_0", "t_proba_1", "t_proba_2", "t_proba_3", "t_proba_4"]
feat_cols = cat_cols + num_cols

concat_df = pd.merge(concat_df, bert_df)

te_pred = pd.read_csv("i18_inference/submission.csv")
val_pred = pd.read_csv("i18_inference/val_pred.csv")
val2_pred = pd.read_csv("i18_inference/kfold_from_2020_to_2021_06_val_pred.csv")

te_pred.columns = ["ncode", "k_proba_0", "k_proba_1", "k_proba_2", "k_proba_3", "k_proba_4"]
val_pred.columns = te_pred.columns
val2_pred.columns = te_pred.columns
val_pred.iloc[:, 1:] = softmax(np.array(val_pred.iloc[:, 1:]), axis=1)
val2_pred.iloc[:, 1:] = softmax(np.array(val2_pred.iloc[:, 1:]), axis=1)
bert_df = pd.concat([te_pred, val_pred, val2_pred], axis=0).reset_index(drop=True)

num_cols += ["k_proba_0", "k_proba_1", "k_proba_2", "k_proba_3", "k_proba_4"]
feat_cols = cat_cols + num_cols

concat_df = pd.merge(concat_df, bert_df)

te_pred = pd.read_csv("i41_inference/submission.csv")
val_pred = pd.read_csv("i41_inference/val_pred.csv")
val2_pred = pd.read_csv("i41_inference/kfold_2021_06_val_pred.csv")
te_pred.columns = ["ncode", "n48_t_proba_0", "n48_t_proba_1", "n48_t_proba_2", "n48_t_proba_3", "n48_t_proba_4"]
val_pred.columns = te_pred.columns
val2_pred.columns = te_pred.columns
val_pred.iloc[:, 1:] = softmax(np.array(val_pred.iloc[:, 1:]), axis=1)
val2_pred.iloc[:, 1:] = softmax(np.array(val2_pred.iloc[:, 1:]), axis=1)
bert_df = pd.concat([te_pred, val_pred, val2_pred], axis=0).reset_index(drop=True)

num_cols += ["n48_t_proba_0", "n48_t_proba_1", "n48_t_proba_2", "n48_t_proba_3", "n48_t_proba_4"]
feat_cols = cat_cols + num_cols

concat_df = pd.merge(concat_df, bert_df)

te_pred = pd.read_csv("i42_inference/submission.csv")
val_pred = pd.read_csv("i42_inference/val_pred.csv")
val2_pred = pd.read_csv("i42_inference/kfold_2021_06_val_pred.csv")
te_pred.columns = ["ncode", "n48_s_proba_0", "n48_s_proba_1", "n48_s_proba_2", "n48_s_proba_3", "n48_s_proba_4"]
val_pred.columns = te_pred.columns
val2_pred.columns = te_pred.columns
val_pred.iloc[:, 1:] = softmax(np.array(val_pred.iloc[:, 1:]), axis=1)
val2_pred.iloc[:, 1:] = softmax(np.array(val2_pred.iloc[:, 1:]), axis=1)
bert_df = pd.concat([te_pred, val_pred, val2_pred], axis=0).reset_index(drop=True)

num_cols += ["n48_s_proba_0", "n48_s_proba_1", "n48_s_proba_2", "n48_s_proba_3", "n48_s_proba_4"]
feat_cols = cat_cols + num_cols

concat_df = pd.merge(concat_df, bert_df)

te_pred = pd.read_csv("i43_inference/submission.csv")
val_pred = pd.read_csv("i43_inference/val_pred.csv")
val2_pred = pd.read_csv("i43_inference/kfold_2021_06_val_pred.csv")
te_pred.columns = ["ncode", "n48_k_proba_0", "n48_k_proba_1", "n48_k_proba_2", "n48_k_proba_3", "n48_k_proba_4"]
val_pred.columns = te_pred.columns
val2_pred.columns = te_pred.columns
val_pred.iloc[:, 1:] = softmax(np.array(val_pred.iloc[:, 1:]), axis=1)
val2_pred.iloc[:, 1:] = softmax(np.array(val2_pred.iloc[:, 1:]), axis=1)
bert_df = pd.concat([te_pred, val_pred, val2_pred], axis=0).reset_index(drop=True)

num_cols += ["n48_k_proba_0", "n48_k_proba_1", "n48_k_proba_2", "n48_k_proba_3", "n48_k_proba_4"]
feat_cols = cat_cols + num_cols

concat_df = pd.merge(concat_df, bert_df)

concat_df.keyword[concat_df.keyword.isnull()] = "None"
concat_df["count_keyword"] = concat_df.apply(count_keyword, axis=1)
num_cols += ["count_keyword"]

concat_df = processing_ncode(concat_df)
num_cols += ['ncode_num']

df = pd.read_csv(Config.train_dir + "/train_stratify.csv")

df = pd.read_csv(Config.train_dir + "/train_stratify.csv")
df = pd.concat([df, test_df])

concat_df["novel_count"] = pd.DataFrame(np.array([0 for _ in range(len(concat_df))]))
d = dict(df['userid'].value_counts(dropna=False))
for i in range(len(concat_df)):
    concat_df["novel_count"][i] = d[concat_df.userid[i]]
num_cols += ["novel_count"]

train_df = pd.concat([train_df, train2_df, test_df]).reset_index(drop=True)


train_df["count_nn"] = train_df.apply(count_nn_story, axis=1).reset_index(drop=True)
concat_df = pd.merge(concat_df, train_df.loc[:, ["ncode", "count_nn"]])
num_cols += ["count_nn"]


train_df["count_n"] = train_df.apply(count_n_story, axis=1).reset_index(drop=True)
concat_df = pd.merge(concat_df, train_df.loc[:, ["ncode", "count_n"]])
num_cols += ["count_n"]

concat_df["biggenre_count"] = pd.DataFrame(np.array([0 for _ in range(len(concat_df))]))
d = dict(concat_df['biggenre'].value_counts(dropna=False))
for i in range(len(concat_df)):
    concat_df["biggenre_count"][i] = d[concat_df.biggenre[i]]
num_cols += ["biggenre_count"]


concat_df = concat_df.drop_duplicates().reset_index(drop=True)

feat_cols = cat_cols + num_cols

all_preds = []
all_val_preds = []
acc = []
score = []
for i in range(5):
    train_df = concat_df[concat_df["fold"] != i]
    train_df = train_df[train_df["fold"] != 6]
    val_df = concat_df[concat_df["fold"] == i]
    test_df = concat_df[concat_df["fold"] == 6]
    print(train_df.shape, val_df.shape, test_df.shape)

    train_x = train_df[feat_cols]
    train_y = train_df[TARGET]
    val_x = val_df[feat_cols]
    val_y = val_df[TARGET]
    test_x = test_df[feat_cols]
    test_y = test_df[TARGET]

    SEED = 0
    model = cb.CatBoostClassifier()
    model.load_model(Config.model_dir + f'/best_model_{i}')

    train_data = cb.Pool(train_x, train_y, cat_features=cat_cols)
    val_data = cb.Pool(val_x, val_y, cat_features=cat_cols)

    val_pred = model.predict(val_x)
    accuracy = sum(val_y == np.round(val_pred)) / len(val_y)
    print(accuracy)
    test_pred = list(logistic.cdf(model.predict(test_x, prediction_type='RawFormulaVal')))
    all_preds.append(test_pred)
    all_val_preds += list(logistic.cdf(model.predict(val_x, prediction_type='RawFormulaVal')))

all_val_df = pd.DataFrame()
for i in range(5):
    val_df = concat_df[concat_df["fold"] == i]
    all_val_df = pd.concat([all_val_df, val_df.ncode])
all_val_df = all_val_df.reset_index(drop=True)

all_val_df["score"] = all_val_preds

all_val_df.columns = ["ncode", "score"]

all_val_df.to_csv(Config.output_dir + "/valid.csv", index=False)

try:
    print(np.bincount(np.round(val_pred).astype(int)))
except:
    print("error")

all_preds = np.array(all_preds)
m_preds = all_preds.mean(0)

sub_df = sub_df.iloc[:, :2]
sub_df.iloc[:, 1] = m_preds
sub_df.columns = ["ncode", "score"]
sub_df.to_csv(Config.output_dir + '/test_submission.csv', index=False)
