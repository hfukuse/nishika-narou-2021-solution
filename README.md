# 作成中
# Nishika 小説家になろうブクマ数予測 ”伸びる”タイトルとは？ ソースコード
https://www.nishika.com/competitions/21/summary  

# Approach  
![スクリーンショット 2021-12-08 13 39 57](https://user-images.githubusercontent.com/61064493/145149949-e105ff33-a635-4ea4-b198-aee2ac775e67.png)
※modelは全て5fold
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
git clone https://github.com/hfukuse/nishika-narou-2021-1st-place-solution.git
```
(※zipファイルを用いる場合はここより下の手順はnishika-narou-2021-1st-place-solutionフォルダーの一つ上の階層で実行してください。)

## 学習済みモデルを使用して予測のみ行う場合  
学習済みモデルを./nishika-narou-2021-1st-place-solution/modelsに設置してください。  
```modelのダウンロード
pip install gdown
sh ./nishika-narou-2021-1st-place-solution/models/get_model.sh
```  
  
```予測
sh ./nishika-narou-2021-1st-place-solution/run_inference_using_trained_models.sh
```
  
## モデルの訓練と予測を行う場合  
```bash
sh ./nishika-narou-2021-1st-place-solution/run.sh
```
  
# その他(環境や注意事項)
OS:Ubuntu 20.04.3    
2 CPU cores  
NVIDIA TESLA P100 GPU × 1 (13 GB of RAM)  
(kaggle notebook環境で動作確認済みです。localで行う場合はkaggle notebook環境に準じた環境で動かしてください。)  
seed:42  
新たにmodelの訓練を行って作成する場合、元々modelsフォルダーに存在するmodelを上書きするので注意してください。  
  
# 参考  
1)CLRP-Pretrain  
https://www.kaggle.com/chamecall/clrp-pretrain  
2)CLRP-Finetune(roberta-large)  
https://www.kaggle.com/chamecall/clrp-finetune-roberta-large  
3)小説家になろう  
https://syosetu.com/  
4)Kaggle-PANDA-1st-place-solution  
https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution  
5)Nishika 財務・非財務情報を活用した株主価値予測 2位ソースコード  
https://github.com/upura/nishika-yuho  
