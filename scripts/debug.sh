SCRIPT_DIR=./nishika-narou-2021-solution/scripts

#pretrain_modelを作成
python ${SCRIPT_DIR}/train/pretrain.py --debug

#create bert feature
#i8_modelを作成
python ${SCRIPT_DIR}/train/n8_nishika_narou_for_train.py --debug
#i9_modelを作成
python ${SCRIPT_DIR}/train/n9_nishika_narou_for_train.py --debug
#i18_modelを作成
python ${SCRIPT_DIR}/train/n18_nishika_narou_for_train.py --debug
#i41_modelを作成
python ${SCRIPT_DIR}/train/n41_nishika_narou_for_train.py --debug
#i42_modelを作成
python ${SCRIPT_DIR}/train/n42_nishika_narou_for_train.py --debug
#i43_modelを作成
python ${SCRIPT_DIR}/train/n43_nishika_narou_for_train.py --debug
#i8_inferenceを作成
python ${SCRIPT_DIR}/inference/i8_nishika_narou_for_inference_title.py
#i9_inferenceを作成
python ${SCRIPT_DIR}/inference/i9_nishika_narou_for_inference.py
#i18_inferenceを作成
python ${SCRIPT_DIR}/inference/i18_nishika_narou_keyword_for_inference.py
#i41_inferenceを作成
python ${SCRIPT_DIR}/inference/i41_nishika_narou_title_for_inference.py
#i42_inferenceを作成
python ${SCRIPT_DIR}/inference/i42_nishika_narou_for_inference.py
#i43_inferenceを作成
python ${SCRIPT_DIR}/inference/i43_nishika_narou_keyword_for_inference.py


#create catboost feature
#n86_modelを作成
python ${SCRIPT_DIR}/train/n86_nishika_narou_for_train.py --debug
#n95_modelを作成
python ${SCRIPT_DIR}/train/n95_nishika_narou_for_train.py --debug
#n95_01_modelを作成
python ${SCRIPT_DIR}/train/n95_nishika_narou_01_for_train.py --debug
#n102_modelを作成
python ${SCRIPT_DIR}/train/n102_nishika_narou_for_train.py --debug
#n107_modelを作成
python ${SCRIPT_DIR}/train/n107_nishika_narou_for_train.py --debug
#n86_inferenceを作成
python ${SCRIPT_DIR}/inference/n86-nishika-narou-use_rmse_for_inference.py
#n95_inferenceを作成
python ${SCRIPT_DIR}/inference/n95-nishika-narou-use_rmse_for_inference.py
#n95_01_inferenceを作成
python ${SCRIPT_DIR}/inference/n95-nishika-narou-use_rmse_01_for_inference.py
#n102_inferenceを作成
python ${SCRIPT_DIR}/inference/n102-nishika-narou-use_rmse_for_inference.py
#n107_inferenceを作成
python ${SCRIPT_DIR}/inference/n107-nishika-narou-use_rmse_for_inference.py

#create model1
#n117_modelを作成
python ${SCRIPT_DIR}/train/n117_nishika_narou_for_train.py
#n117_inferenceを作成
python ${SCRIPT_DIR}/inference/n117-nishika-narou-change-multirmse_for_inference.py

#copy model
cp -r ../input/all-model-nishika-narou/* ./nishika-narou-2021-solution/models/

#predict using final model
python ${SCRIPT_DIR}/inference/predict.py