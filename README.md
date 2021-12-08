# 作成中
# Nishika 小説家になろうブクマ数予測 ”伸びる”タイトルとは？ ソースコード
https://www.nishika.com/competitions/21/summary  

# Approach  
![スクリーンショット 2021-12-08 13 39 57](https://user-images.githubusercontent.com/61064493/145149949-e105ff33-a635-4ea4-b198-aee2ac775e67.png)
# Input Data  
```
data
└── input
    └── nishika-narou
        └── train.csv
        |
        └── test.csv
        |
        └── sample_submission.csv
```

# Run  
## はじめに  
```bash
$ git clone https://github.com/hfukuse/nishika-narou-2021-1st-place-solution.git
```
(※zipファイルを用いる場合はここより下の手順はnishika-narou-2021-1st-place-solutionフォルダーの一つ上の階層で実行してください。)

## 学習済みモデルを使用して予測のみ行う場合  
```bash
$ sh ./nishika-narou-2021-1st-place-solution/run_inference_using_trained_models.sh
```
  
## モデルの訓練と予測を行う場合  
```bash
$ sh ./nishika-narou-2021-1st-place-solution/run.sh
```
  
## その他(環境など)  
2 CPU cores  
NVIDIA TESLA P100 GPU × 1 (13 GB of RAM)  
OS:Ubuntu 20.04.3  
(kaggle notebook環境で動作確認済みです。localで行う場合はkaggle notebook環境に準じた環境で動かしてください。)  
seed:42  
新たにmodelの訓練を行って作成する場合、元々modelsフォルダーに存在するmodelを上書きするので注意してください。  
