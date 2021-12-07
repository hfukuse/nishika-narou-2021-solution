import os

import catboost as cb
import numpy as np
import pandas as pd
from scipy.stats import logistic
from scipy.special import softmax
import re
import sklearn.preprocessing as sp
import json
import sys
import argparse


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--debug", action="store_true", help="debug")
    arg("--settings", default="./settings.json", type=str, help="settings path")
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
    output_dir = js["n117"]["output_dir"]
    model_dir = js["models_dir"]+"/"+js["n117"]["model_dir"]
    narou_dir = js["narou_dir"]
    i8_inf=js["i8"]["output_dir"]
    i9_inf=js["i9"]["output_dir"]
    i18_inf=js["i18"]["output_dir"]
    i41_inf=js["i41"]["output_dir"]
    i42_inf=js["i42"]["output_dir"]
    i43_inf=js["i43"]["output_dir"]
    n86_inf=js["n86"]["output_dir"]
    n95_inf=js["n95"]["output_dir"]
    n95_01_inf=js["n95_01"]["output_dir"]
    n102_inf=js["n102"]["output_dir"]
    n107_inf=js["n107"]["output_dir"]


sys.path.append(Config.narou_dir)
sys.path.append(os.path.basename(__file__)+"/..")
from utils.preprocess import remove_url,processing_ncode,count_keyword,count_nn_story,count_n_story

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
    past_days = raw_df[raw_df.userid == j].past_days
    for i in range(len(past_days) - 1):
        past_days_from_previous_work.append(past_days.iloc[i] - past_days.iloc[i + 1])
    raw_df.loc[raw_df.userid == j, "past_days_from_previous_work"] = past_days_from_previous_work

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

concat_df = pd.concat([raw_df, tags_df], axis=1)
concat_df.shape

num_cols = ['userid', 'past_days', 'title_length', 'length']

cat_cols = ['writer', 'biggenre', 'genre', 'novel_type', 'isr15', 'isbl', 'isgl', 'iszankoku', "tenni_tennsei",
            'pc_or_k'] + key_w
num_cols += ["past_days_from_previous_work"]
num_cols += ["genre_each_count", "biggenre_each_count"]
num_cols += ["mean_fav", "sum_fav"]
num_cols += ["tanpen_each_count", "tyohen_each_count"]
num_cols += ["past_days_from_previous_work_tyohen", "past_days_from_previous_work_tanpen"]

feat_cols = cat_cols + num_cols

ID = 'ncode'
TARGET = 'fav_novel_cnt_bin'

te_pred = pd.read_csv(Config.i9_inf+"/submission.csv")
val_pred = pd.read_csv(Config.i9_inf+"/val_pred.csv")
val2_pred = pd.read_csv(Config.i9_inf+"/kfold_from_2020_to_2021_06_val_pred.csv")
val_pred.columns = te_pred.columns
val2_pred.columns = te_pred.columns
val_pred.iloc[:, 1:] = softmax(np.array(val_pred.iloc[:, 1:]), axis=1)
val2_pred.iloc[:, 1:] = softmax(np.array(val2_pred.iloc[:, 1:]), axis=1)
bert_df = pd.concat([te_pred, val_pred, val2_pred], axis=0).reset_index(drop=True)

num_cols += list(bert_df.columns)[1:]
feat_cols = cat_cols + num_cols

concat_df = pd.merge(concat_df, bert_df)

te_pred = pd.read_csv(Config.i8_inf+"/submission.csv")
val_pred = pd.read_csv(Config.i8_inf+"/val_pred.csv")
val2_pred = pd.read_csv(Config.i8_inf+"/kfold_from_2020_to_2021_06_val_pred.csv")

te_pred.columns = ["ncode", "t_proba_0", "t_proba_1", "t_proba_2", "t_proba_3", "t_proba_4"]
val_pred.columns = te_pred.columns
val2_pred.columns = te_pred.columns
val_pred.iloc[:, 1:] = softmax(np.array(val_pred.iloc[:, 1:]), axis=1)
val2_pred.iloc[:, 1:] = softmax(np.array(val2_pred.iloc[:, 1:]), axis=1)
bert_df = pd.concat([te_pred, val_pred, val2_pred], axis=0).reset_index(drop=True)

num_cols += ["t_proba_0", "t_proba_1", "t_proba_2", "t_proba_3", "t_proba_4"]
feat_cols = cat_cols + num_cols

concat_df = pd.merge(concat_df, bert_df)

te_pred = pd.read_csv(Config.i18_inf+"/submission.csv")
val_pred = pd.read_csv(Config.i18_inf+"/val_pred.csv")
val2_pred = pd.read_csv(Config.i18_inf+"/kfold_from_2020_to_2021_06_val_pred.csv")

te_pred.columns = ["ncode", "k_proba_0", "k_proba_1", "k_proba_2", "k_proba_3", "k_proba_4"]
val_pred.columns = te_pred.columns
val2_pred.columns = te_pred.columns
val_pred.iloc[:, 1:] = softmax(np.array(val_pred.iloc[:, 1:]), axis=1)
val2_pred.iloc[:, 1:] = softmax(np.array(val2_pred.iloc[:, 1:]), axis=1)
bert_df = pd.concat([te_pred, val_pred, val2_pred], axis=0).reset_index(drop=True)

num_cols += ["k_proba_0", "k_proba_1", "k_proba_2", "k_proba_3", "k_proba_4"]
feat_cols = cat_cols + num_cols

concat_df = pd.merge(concat_df, bert_df)

te_pred = pd.read_csv(Config.i41_inf+"/submission.csv")
val_pred = pd.read_csv(Config.i41_inf+"/val_pred.csv")
val2_pred = pd.read_csv(Config.i41_inf+"/kfold_2021_06_val_pred.csv")
te_pred.columns = ["ncode", "n48_t_proba_0", "n48_t_proba_1", "n48_t_proba_2", "n48_t_proba_3", "n48_t_proba_4"]
val_pred.columns = te_pred.columns
val2_pred.columns = te_pred.columns
val_pred.iloc[:, 1:] = softmax(np.array(val_pred.iloc[:, 1:]), axis=1)
val2_pred.iloc[:, 1:] = softmax(np.array(val2_pred.iloc[:, 1:]), axis=1)
bert_df = pd.concat([te_pred, val_pred, val2_pred], axis=0).reset_index(drop=True)

num_cols += ["n48_t_proba_0", "n48_t_proba_1", "n48_t_proba_2", "n48_t_proba_3", "n48_t_proba_4"]
feat_cols = cat_cols + num_cols

concat_df = pd.merge(concat_df, bert_df)

te_pred = pd.read_csv(Config.i42_inf+"/submission.csv")
val_pred = pd.read_csv(Config.i42_inf+"/val_pred.csv")
val2_pred = pd.read_csv(Config.i42_inf+"/kfold_2021_06_val_pred.csv")
te_pred.columns = ["ncode", "n48_s_proba_0", "n48_s_proba_1", "n48_s_proba_2", "n48_s_proba_3", "n48_s_proba_4"]
val_pred.columns = te_pred.columns
val2_pred.columns = te_pred.columns
val_pred.iloc[:, 1:] = softmax(np.array(val_pred.iloc[:, 1:]), axis=1)
val2_pred.iloc[:, 1:] = softmax(np.array(val2_pred.iloc[:, 1:]), axis=1)
bert_df = pd.concat([te_pred, val_pred, val2_pred], axis=0).reset_index(drop=True)

num_cols += ["n48_s_proba_0", "n48_s_proba_1", "n48_s_proba_2", "n48_s_proba_3", "n48_s_proba_4"]
feat_cols = cat_cols + num_cols

concat_df = pd.merge(concat_df, bert_df)

te_pred = pd.read_csv(Config.i43_inf+"/submission.csv")
val_pred = pd.read_csv(Config.i43_inf+"/val_pred.csv")
val2_pred = pd.read_csv(Config.i43_inf+"/kfold_2021_06_val_pred.csv")
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

df1 = pd.read_csv(Config.n86_inf+"/valid.csv")
df2 = pd.read_csv(Config.n86_inf+"/test_submission.csv")
df = pd.concat([df1, df2]).reset_index(drop=True)
df.columns = ["ncode", "rmse"]
concat_df = pd.merge(concat_df, df)
num_cols += ["rmse"]

df1 = pd.read_csv(Config.n95_01_inf+"/valid.csv")
df2 = pd.read_csv(Config.n95_01_inf+"/test_submission.csv")
df = pd.concat([df1, df2]).reset_index(drop=True)
df.columns = ["ncode", "01rmse"]
concat_df = pd.merge(concat_df, df)
num_cols += ["01rmse"]

df1 = pd.read_csv(Config.n95_inf+"/valid.csv")
df2 = pd.read_csv(Config.n95_inf+"/test_submission.csv")
df = pd.concat([df1, df2]).reset_index(drop=True)
df.columns = ["ncode", "binary"]
concat_df = pd.merge(concat_df, df)
num_cols += ["binary"]

df1 = pd.read_csv(Config.n102_inf+"/valid.csv")
df2 = pd.read_csv(Config.n102_inf+"/test_submission.csv")
df = pd.concat([df1, df2]).reset_index(drop=True)
df.columns = ["ncode", "binary3"]
concat_df = pd.merge(concat_df, df)
num_cols += ["binary3"]

df1 = pd.read_csv(Config.n107_inf+"/valid.csv")
df2 = pd.read_csv(Config.n107_inf+"/test_submission.csv")
df = pd.concat([df1, df2]).reset_index(drop=True)
df.columns = ["ncode", "-101rmse"]
concat_df = pd.merge(concat_df, df)
num_cols += ["-101rmse"]

concat_df = concat_df.drop_duplicates().reset_index(drop=True)

import sklearn.preprocessing as sp
enc = sp.OneHotEncoder(categories='auto', sparse=False)
enc.set_params( categories=[[0,1,2,3,4]] )

feat_cols = cat_cols + num_cols

acc = []
score = []

all_val_df = pd.DataFrame()
all_preds = []
all_val_preds = []
for i in range(5):
    train_df = concat_df[concat_df["fold"]!=i]
    train_df = train_df[train_df["fold"]!=6]
#    val_df=concat_df.iloc[:21711]
    val_df = concat_df[concat_df["fold"] == i]
    test_df = concat_df[concat_df["fold"] == 6]

    train_x = train_df[feat_cols]
    train_y = train_df[TARGET]
    val_x = val_df[feat_cols]
    val_y = val_df[TARGET]
    test_x = test_df[feat_cols]
    test_y = test_df[TARGET]
    train_y=pd.DataFrame(enc.fit_transform(pd.DataFrame(train_y.map(int))))
    val_y=pd.DataFrame(enc.fit_transform(pd.DataFrame(val_y.map(int))))

    params = {
        'loss_function': 'MultiRMSE',
        "random_state": 420,
        "num_boost_round": 50000,
        "early_stopping_rounds": 200,
        "task_type": "CPU",
        "use_best_model": True,
        'bagging_temperature': 0
    }

    model = cb.CatBoostRegressor(**params)

    train_data = cb.Pool(train_x, train_y,cat_features=cat_cols)#,text_features=text_cols)
    val_data = cb.Pool(val_x, val_y,cat_features=cat_cols)#,text_features=text_cols)

    model = model.fit(
        train_data,
        eval_set=val_data,
        early_stopping_rounds=200,
        verbose=100
    )

    val_pred = model.predict(val_x)  # predict_proba(val_x)
    val_pred_max = np.argmax(val_pred, axis=1)  # 最尤と判断したクラスの値にする
    accuracy = sum(np.argmax(val_y.values, axis=1) == val_pred_max) / len(val_y)
    print(accuracy)
    test_pred = model.predict(test_x)  # _proba(test_x)
    all_preds.append(test_pred)
    all_val_preds += list(val_pred)
    all_val_df = pd.concat([all_val_df, val_df.ncode])
    model.save_model(Config.model_dir + f'/best_model_{i}')

    acc.append(accuracy)
    score.append(model.best_score_["validation"]["MultiRMSE"])
print("**acc**")
print(np.mean(np.array(acc)))
print("**score**")
print(np.mean(np.array(score)))
all_val_df = all_val_df.reset_index(drop=True)
all_val_df = pd.concat([all_val_df, pd.DataFrame(all_val_preds)], axis=1)
all_val_df.columns = ["ncode"] + ["mr_proba_" + str(i) for i in range(5)]
all_val_df.to_csv(Config.output_dir + "/valid.csv", index=False)

try:
    print(np.bincount(np.round(val_pred).astype(int)))
except:
    print("error")

all_preds = np.array(all_preds)
m_preds = all_preds.mean(0)

sub_df.iloc[:, 1:] = m_preds
sub_df.to_csv(Config.output_dir + '/test_submission.csv', index=False)
