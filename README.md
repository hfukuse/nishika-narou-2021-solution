# 作成中
# Nishika 小説家になろうブクマ数予測 ”伸びる”タイトルとは？ 1位ソースコード
https://www.nishika.com/competitions/21/summary  
本ソースコードとモデルは上記のNishikaによって開催されたコンペティションで提供されたデータセットを元に作成されました。  
提供されたデータには2007年から2021年8月までの間に完結した作品が訓練データとして含まれていました。  
精度を測定するために2021年8月以降に完結した作品についてブックマークが5段階にbin化されたものに対して予測を行いました。指標はloglossで最終スコアは0.627155でした。  
validation用のデータに対するブックマーク度の正答率は72.23%でした。  

# Approach  
![スクリーンショット 2021-12-08 13 39 57](https://user-images.githubusercontent.com/61064493/145149949-e105ff33-a635-4ea4-b198-aee2ac775e67.png)　　
![スクリーンショット 2021-12-10 13 41 47](https://user-images.githubusercontent.com/61064493/145518357-6bbe1364-0768-4526-b1e0-c99415c97d9d.png)　　
　　
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
git clone https://github.com/hfukuse/nishika-narou-2021-solution.git
```
(※zipファイルを用いる場合はここより下の手順はnishika-narou-2021-solutionフォルダーの一つ上の階層で実行してください。)

## 学習済みモデルを使用して予測のみ行う場合  
学習済みモデルを./nishika-narou-2021-solution/modelsに設置してください。  
  
(modelのダウンロード)合計12GBほど  
```
pip install gdown
sh ./nishika-narou-2021-solution/models/get_model.sh
```  
kaggle notebookのoutputファイルの最大容量は19GBほどで,ダウンロードと展開を一つのnotebookでやると容量をオーバーしてしまうので、一度models.zipをダウンロードして専用のnotebookを作成するか、あるいはdatasetを作成するなどの工夫が必要となります。  
  
(予測)  
```
sh ./nishika-narou-2021-solution/run_inference_using_trained_models.sh
```
  
## モデルの訓練と予測を行う場合  
```bash
sh ./nishika-narou-2021-solution/run.sh
```
  
# その他(環境や注意事項)
OS:Ubuntu 20.04.3    
2 CPU cores  
NVIDIA TESLA P100 GPU × 1 (13 GB of RAM)  
(kaggle notebook環境で動作確認済みです。localで行う場合はkaggle notebook環境に準じた環境で動かしてください。)  
seed:42  
新たにmodelの訓練を行って作成する場合、元々modelsフォルダーに存在するmodelを上書きするので注意してください。  
# 簡単なEDA  
![スクリーンショット 2021-12-10 13 43 30](https://user-images.githubusercontent.com/61064493/145518508-433ebb61-3997-475a-9271-2deac73fee36.png)  
  
(上の表の見方)機械学習モデルによってブックマーク度が0と予測された作品のうち1150作品が実際にブックマーク度が0、279作品がブックマーク度が1、6作品がブックマーク度が2、ブックマーク度が0と予測された作品でブックマーク度が3以上の作品はない。...というように解釈してください。
validation用データに対する予想は正答率は72.23%でした  
(正式には公開されていませんが、ブックマーク度0がブックマーク0、ブックマーク度1がブックマーク1-9、ブックマーク度2がブックマーク10-99、ブックマーク度3がブックマーク100-999、ブックマーク度4がブックマーク1000以上に当たると考えられます。)  
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
