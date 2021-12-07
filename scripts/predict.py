import pandas as pd
import numpy as np

from scipy.special import softmax
import catboost as cb
import argparse
import json
import sys



def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--settings", default="./settings.json", type=str, help="settings path")
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
    output_dir = "."
    model_dir = js["models_dir"]
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
    n117_inf=js["n117"]["output_dir"]


sys.path.append(Config.narou_dir)

from utils.preprocess import remove_url,processing_ncode,count_keyword,count_nn_story,count_n_story

#作れる#with posはtrain_val_split dirへの統合はありかも
train_df = pd.read_csv(Config.train_dir+'/kfold_2021_06.csv')
train2_df = pd.read_csv(Config.train_dir+'/kfold_from_2020_to_2021_06.csv')
test_df = pd.read_csv(Config.dataset_dir+'/test.csv')
sub_df = pd.read_csv(Config.dataset_dir+'/sample_submission.csv')

test_df["fold"]=6

concat_df=pd.read_csv("concat_df.csv")
concat_df.shape

cat_cols = ['writer', 'biggenre', 'genre', 'novel_type','isr15', 'isbl', 'isgl', 'iszankoku', "tenni_tennsei", 'pc_or_k']+["ざまあ"]
num_cols = ['userid','past_days','title_length','length']
num_cols += ["past_days_from_previous_work"]
num_cols += ["genre_each_count","biggenre_each_count"]
num_cols += ["mean_fav","sum_fav"]
num_cols += ["tanpen_each_count","tyohen_each_count"]
num_cols += ["past_days_from_previous_work_tyohen","past_days_from_previous_work_tanpen"]

ID = 'ncode'
TARGET = 'fav_novel_cnt_bin'



te_pred=pd.read_csv(Config.i8_inf+"/submission.csv")
val_pred=pd.read_csv(Config.i8_inf+"/val_pred.csv")
val2_pred=pd.read_csv(Config.i8_inf+"/kfold_from_2020_to_2021_06_val_pred.csv")

te_pred.columns=["ncode","t_proba_0","t_proba_1","t_proba_2","t_proba_3","t_proba_4"]
val_pred.columns=te_pred.columns
val2_pred.columns=te_pred.columns
val_pred.iloc[:,1:]=softmax(np.array(val_pred.iloc[:,1:]),axis=1)
val2_pred.iloc[:,1:]=softmax(np.array(val2_pred.iloc[:,1:]),axis=1)
bert_df=pd.concat([te_pred,val_pred,val2_pred],axis=0).reset_index(drop=True)

num_cols += ["t_proba_0","t_proba_1","t_proba_2","t_proba_3","t_proba_4"]
feat_cols=cat_cols+num_cols

concat_df=pd.merge(concat_df,bert_df)



te_pred=pd.read_csv(Config.i9_inf+"/submission.csv")
val_pred=pd.read_csv(Config.i9_inf+"/val_pred.csv")
val2_pred=pd.read_csv(Config.i9_inf+"/kfold_from_2020_to_2021_06_val_pred.csv")

val_pred.columns=te_pred.columns
val2_pred.columns=te_pred.columns
val_pred.iloc[:,1:]=softmax(np.array(val_pred.iloc[:,1:]),axis=1)
val2_pred.iloc[:,1:]=softmax(np.array(val2_pred.iloc[:,1:]),axis=1)
bert_df=pd.concat([te_pred,val_pred,val2_pred],axis=0).reset_index(drop=True)

num_cols += list(bert_df.columns)[1:]
feat_cols=cat_cols+num_cols

concat_df=pd.merge(concat_df,bert_df)


te_pred=pd.read_csv(Config.i18_inf+"/submission.csv")
val_pred=pd.read_csv(Config.i18_inf+"/val_pred.csv")
val2_pred=pd.read_csv(Config.i18_inf+"/kfold_from_2020_to_2021_06_val_pred.csv")

te_pred.columns=["ncode","k_proba_0","k_proba_1","k_proba_2","k_proba_3","k_proba_4"]
val_pred.columns=te_pred.columns
val2_pred.columns=te_pred.columns
val_pred.iloc[:,1:]=softmax(np.array(val_pred.iloc[:,1:]),axis=1)
val2_pred.iloc[:,1:]=softmax(np.array(val2_pred.iloc[:,1:]),axis=1)
bert_df=pd.concat([te_pred,val_pred,val2_pred],axis=0).reset_index(drop=True)

num_cols += ["k_proba_0","k_proba_1","k_proba_2","k_proba_3","k_proba_4"]
feat_cols=cat_cols+num_cols

concat_df=pd.merge(concat_df,bert_df)


te_pred=pd.read_csv(Config.i41_inf+"/submission.csv")
val_pred=pd.read_csv(Config.i41_inf+"/val_pred.csv")
val2_pred=pd.read_csv(Config.i41_inf+"/kfold_2021_06_val_pred.csv")
te_pred.columns=["ncode","n48_t_proba_0","n48_t_proba_1","n48_t_proba_2","n48_t_proba_3","n48_t_proba_4"]
val_pred.columns=te_pred.columns
val2_pred.columns=te_pred.columns
val_pred.iloc[:,1:]=softmax(np.array(val_pred.iloc[:,1:]),axis=1)
val2_pred.iloc[:,1:]=softmax(np.array(val2_pred.iloc[:,1:]),axis=1)
bert_df=pd.concat([te_pred,val_pred,val2_pred],axis=0).reset_index(drop=True)

num_cols += ["n48_t_proba_0","n48_t_proba_1","n48_t_proba_2","n48_t_proba_3","n48_t_proba_4"]
feat_cols=cat_cols+num_cols

concat_df=pd.merge(concat_df,bert_df)


te_pred=pd.read_csv(Config.i42_inf+"/submission.csv")
val_pred=pd.read_csv(Config.i42_inf+"/val_pred.csv")
val2_pred=pd.read_csv(Config.i42_inf+"/kfold_2021_06_val_pred.csv")
te_pred.columns=["ncode","n48_s_proba_0","n48_s_proba_1","n48_s_proba_2","n48_s_proba_3","n48_s_proba_4"]
val_pred.columns=te_pred.columns
val2_pred.columns=te_pred.columns
val_pred.iloc[:,1:]=softmax(np.array(val_pred.iloc[:,1:]),axis=1)
val2_pred.iloc[:,1:]=softmax(np.array(val2_pred.iloc[:,1:]),axis=1)
bert_df=pd.concat([te_pred,val_pred,val2_pred],axis=0).reset_index(drop=True)

num_cols += ["n48_s_proba_0","n48_s_proba_1","n48_s_proba_2","n48_s_proba_3","n48_s_proba_4"]
feat_cols=cat_cols+num_cols

concat_df=pd.merge(concat_df,bert_df)



te_pred=pd.read_csv(Config.i43_inf+"/submission.csv")
val_pred=pd.read_csv(Config.i43_inf+"/val_pred.csv")
val2_pred=pd.read_csv(Config.i43_inf+"/kfold_2021_06_val_pred.csv")
te_pred.columns=["ncode","n48_k_proba_0","n48_k_proba_1","n48_k_proba_2","n48_k_proba_3","n48_k_proba_4"]
val_pred.columns=te_pred.columns
val2_pred.columns=te_pred.columns
val_pred.iloc[:,1:]=softmax(np.array(val_pred.iloc[:,1:]),axis=1)
val2_pred.iloc[:,1:]=softmax(np.array(val2_pred.iloc[:,1:]),axis=1)
bert_df=pd.concat([te_pred,val_pred,val2_pred],axis=0).reset_index(drop=True)

num_cols += ["n48_k_proba_0","n48_k_proba_1","n48_k_proba_2","n48_k_proba_3","n48_k_proba_4"]
feat_cols=cat_cols+num_cols

concat_df=pd.merge(concat_df,bert_df)

concat_df.keyword[concat_df.keyword.isnull()]="None"
concat_df["count_keyword"]=concat_df.apply(count_keyword,axis=1)
num_cols+=["count_keyword"]


concat_df=processing_ncode(concat_df)
num_cols+=['ncode_num']


df=pd.read_csv(Config.train_dir+"/train_stratify.csv")
df=pd.concat([df,test_df])

concat_df["novel_count"]=pd.DataFrame(np.array([0 for _ in range(len(concat_df))]))
d=dict(df['userid'].value_counts(dropna=False))
for i in range(len(concat_df)):
    concat_df["novel_count"][i]=d[concat_df.userid[i]]
num_cols+=["novel_count"]

train_df=pd.concat([train_df,train2_df,test_df]).reset_index(drop=True)


train_df["count_nn"]=train_df.apply(count_nn_story,axis=1).reset_index(drop=True)
concat_df=pd.merge(concat_df,train_df.loc[:,["ncode","count_nn"]])
num_cols+=["count_nn"]

train_df["count_n"]=train_df.apply(count_n_story,axis=1).reset_index(drop=True)
concat_df=pd.merge(concat_df,train_df.loc[:,["ncode","count_n"]])
num_cols+=["count_n"]


concat_df["biggenre_count"]=pd.DataFrame(np.array([0 for _ in range(len(concat_df))]))
d=dict(concat_df['biggenre'].value_counts(dropna=False))
for i in range(len(concat_df)):
    concat_df["biggenre_count"][i]=d[concat_df.biggenre[i]]
num_cols+=["biggenre_count"]


concat_df=concat_df.drop_duplicates().reset_index(drop=True)


df1=pd.read_csv("n86_inference/valid.csv")
df2=pd.read_csv("n86_inference/test_submission.csv")
df=pd.concat([df1,df2]).reset_index(drop=True)
df.columns=["ncode","rmse"]
concat_df=pd.merge(concat_df,df)
num_cols+=["rmse"]


df1=pd.read_csv("n95_inference/valid.csv")
df2=pd.read_csv("n95_inference/test_submission.csv")
df=pd.concat([df1,df2]).reset_index(drop=True)
df.columns=["ncode","binary"]
concat_df=pd.merge(concat_df,df)
num_cols+=["binary"]


df1=pd.read_csv("n95_01_inference/valid.csv")
df2=pd.read_csv("n95_01_inference/test_submission.csv")
df=pd.concat([df1,df2]).reset_index(drop=True)
df.columns=["ncode","01rmse"]
concat_df=pd.merge(concat_df,df)
num_cols+=["01rmse"]


df1=pd.read_csv("n102_inference/valid.csv")
df2=pd.read_csv("n102_inference/test_submission.csv")
df=pd.concat([df1,df2]).reset_index(drop=True)
df.columns=["ncode","binary3"]
concat_df=pd.merge(concat_df,df)
num_cols+=["binary3"]


df1=pd.read_csv("n107_inference/valid.csv")
df2=pd.read_csv("n107_inference/test_submission.csv")
df=pd.concat([df1,df2]).reset_index(drop=True)
df.columns=["ncode","-101rmse"]
concat_df=pd.merge(concat_df,df)
num_cols+=["-101rmse"]


df1=pd.read_csv("n117_inference/valid.csv")
df2=pd.read_csv("n117_inference/test_submission.csv")
df2.columns=df1.columns
df=pd.concat([df1,df2]).reset_index(drop=True)
df.columns=["ncode","mr_proba_0","mr_proba_1","mr_proba_2","mr_proba_3","mr_proba_4"]
df["mr_proba"]=df["mr_proba_0"]*5000+df["mr_proba_1"]*500+df["mr_proba_2"]*50+df["mr_proba_3"]*5+df["mr_proba_4"]*0
concat_df=pd.merge(concat_df,df)
num_cols+=["mr_proba"]

text_cols = []
feat_cols = cat_cols + num_cols

all_preds = []
all_val_preds = []
acc = []
score = []
for i in range(5):
    train_df = concat_df[concat_df["fold"] != i]
    train_df = train_df[train_df["fold"] != 6]
    train2_df = concat_df.iloc[21711:]
    train2_df = train2_df[train2_df["fold"] == i]

    train_df = pd.concat([train_df, train2_df])
    val_df = concat_df.iloc[:21711]
    val_df = val_df[val_df["fold"] == i]
    test_df = concat_df[concat_df["fold"] == 6]
    print(train_df.shape, val_df.shape, test_df.shape)

    train_x = train_df[feat_cols]
    train_y = train_df[TARGET]
    val_x = val_df[feat_cols]
    val_y = val_df[TARGET]
    test_x = test_df[feat_cols]
    test_y = test_df[TARGET]
    train_x.shape

    SEED = 0

    model = cb.CatBoostClassifier()
    model.load_model(Config.model_dir + f'/n126_model/best_model_{i}')

    train_data = cb.Pool(train_x, train_y, cat_features=cat_cols)
    val_data = cb.Pool(val_x, val_y, cat_features=cat_cols)

    val_pred = model.predict_proba(val_x)
    val_pred_max = np.argmax(val_pred, axis=1)
    accuracy = sum(val_y == val_pred_max) / len(val_y)
    print(accuracy)
    test_pred = model.predict_proba(test_x)
    all_preds.append(test_pred)
    all_val_preds += list(val_pred)

    acc.append(accuracy)
print("**acc**")
print(np.mean(np.array(acc)))
text_cols=[]
feat_cols = cat_cols + num_cols


all_preds=[]
all_val_preds=[]
acc=[]
score=[]
for i in range(5):
    train_df = concat_df[concat_df["fold"]!=i]
    train_df = train_df[train_df["fold"]!=6]
    train2_df=concat_df.iloc[21711:]
    train2_df=train2_df[train2_df["fold"]==i]

    train_df=pd.concat([train_df,train2_df])
    val_df=concat_df.iloc[:21711]
    val_df = val_df[val_df["fold"]==i]
    test_df = concat_df[concat_df["fold"]==6]
    print(train_df.shape, val_df.shape, test_df.shape)

    train_x = train_df[feat_cols]
    train_y = train_df[TARGET]
    val_x = val_df[feat_cols]
    val_y = val_df[TARGET]
    test_x = test_df[feat_cols]
    test_y = test_df[TARGET]
    train_x.shape


    SEED = 0

    model = cb.CatBoostClassifier()
    model.load_model(Config.model_dir+f'/n126_model/best_model_{i}')

    train_data = cb.Pool(train_x, train_y,cat_features=cat_cols)
    val_data = cb.Pool(val_x, val_y,cat_features=cat_cols)


    val_pred = model.predict_proba(val_x)
    val_pred_max = np.argmax(val_pred, axis=1)
    accuracy = sum(val_y == val_pred_max) / len(val_y)
    print(accuracy)
    test_pred =model.predict_proba(test_x)
    all_preds.append(test_pred)
    all_val_preds+=list(val_pred)

    acc.append(accuracy)
print("**acc**")
print(np.mean(np.array(acc)))

all_preds=np.array(all_preds)
m_preds=all_preds.mean(0)
sub_df.iloc[:, 1:] = m_preds
sub_df.to_csv('test_submission.csv', index=False)