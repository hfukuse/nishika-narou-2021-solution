# 変数の設定
```
SCRIPT_DIR=./nishika-narou-2021-1st-place-solution/scripts
```
# 前処理
```
sh ${SCRIPT_DIR}/preprocess.sh
```
・dataset_dirからデータを読み込む.  
・訓練データを2020年から2021年6月までと2021年6月以降に分けて、訓練データとtestデータを5foldで分けて、いくつかのfeature engineering行う.  
・train_dirにtrainの5foldのデータを出力し、pos_dirにfeature engineering結果をconcatしたデータを出力する.  
  
# 訓練と予測
## bert
```
sh ${SCRIPT_DIR}/train_bert.sh
```
・dataset_dirとtrain_dirからデータを読み込む.  
・まず、読み込んだデータを元にbertのpretrainを行う。次にそのpretrain modelを用いて、6つのモデルを作成する.  
※(2020年から2021年6月、2021年6月以降)のどちらを学習データとするか
(title、story、keyword)のどれを学習データとするかの組み合わせで2×3=6モデル  
・models_dirにモデルが含まれたディレクトリを出力する.  

```
sh ${SCRIPT_DIR}/predict_bert.sh
```
・models_dirからmodelとtrain_dirからデータを読み込む.  
・bertモデルを用いて後のモデルで特徴量として用いるための予測を行う.  
・output_dirに予測結果が含まれたディレクトリを出力する.  
  
## catboost
```
sh ${SCRIPT_DIR}/train_catboost.sh
```
・train_dirとdataset_dirとpos_dirとoutput_dirからデータを読み込む.  
・catboostモデルを作成し、後で特徴量として用いるための予測を行う.  
・models_dirにモデルを出力し、output_dirに予測結果が含まれたディレクトリを出力する.  
  
```
sh ${SCRIPT_DIR}/predict_catboost.sh
```
・train_dirとdataset_dirとpos_dirとoutput_dirからデータを読み込み、models_dirからモデルを読み込む.  
・catboostを用いて後のモデルで特徴量として用いるための予測を行う.  
・output_dirに予測結果が含まれたディレクトリを出力する.  
  
## model1  
```
sh ${SCRIPT_DIR}/train_model1.sh
```
・train_dirとdataset_dirとpos_dirとoutput_dirからデータを読み込む.  
・catboostモデルを作成し、後で特徴量として用いるための予測を行う.  
・models_dirにモデルを出力し、output_dirに予測結果が含まれたディレクトリを出力する.  
  
```
sh ${SCRIPT_DIR}/predict_model1.sh
```
・train_dirとdataset_dirとpos_dirとoutput_dirからデータを読み込み、models_dirからモデルを読みこむ.  
・catboostを用いて後のモデルで特徴量として用いるための予測を行う.  
・output_dirに予測結果が含まれたディレクトリを出力する.  
  
## 最終モデル  
```
sh ${SCRIPT_DIR}/train_model2.sh
```
・train_dirとdataset_dirとpos_dirとoutput_dirからデータを読み込む.  
・catboostモデルを作成する.  
・models_dirにモデルを出力する。  

  
```
sh ${SCRIPT_DIR}/predict_model2.sh
```
・train_dirとdataset_dirとpos_dirとoutput_dirからデータとmodels_dirからモデルを読み込む.  
・catboostモデルを用いて最終予測を行う.  
・result_dirに予測結果が含まれたディレクトリを出力する.  

