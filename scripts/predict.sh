NAROU_DIR=./nishika-narou-2021-1st-place-solution

#i8_inferenceを作成
python ${NAROU_DIR}/scripts_for_kaggle/i8_nishika_narou_for_inference_title.py
#i9_inferenceを作成
python ${NAROU_DIR}/scripts_for_kaggle/i9_nishika_narou_for_inference.py
#i18_inferenceを作成
python ${NAROU_DIR}/scripts_for_kaggle/i18_nishika_narou_keyword_for_inference.py
#i41_inferenceを作成
python ${NAROU_DIR}/scripts_for_kaggle/i41_nishika_narou_title_for_inference.py
#i42_inferenceを作成
python ${NAROU_DIR}/scripts_for_kaggle/i42_nishika_narou_for_inference.py
#i43_inferenceを作成
python ${NAROU_DIR}/scripts_for_kaggle/i43_nishika_narou_keyword_for_inference.py


#n86_inferenceを作成
python ${NAROU_DIR}/scripts_for_kaggle/n86-nishika-narou-use_rmse_for_inference.py
#n95_inferenceを作成
python ${NAROU_DIR}/scripts_for_kaggle/n95-nishika-narou-use_rmse_for_inference.py
#n95_01_inferenceを作成
python ${NAROU_DIR}/scripts_for_kaggle/n95-nishika-narou-use_rmse_01_for_inference.py
#n102_inferenceを作成
python ${NAROU_DIR}/scripts_for_kaggle/n102-nishika-narou-use_rmse_for_inference.py
#n107_inferenceを作成
python ${NAROU_DIR}/scripts_for_kaggle/n107-nishika-narou-use_rmse_for_inference.py
#n117_inferenceを作成
python ${NAROU_DIR}/scripts_for_kaggle/n117-nishika-narou-change-multirmse_for_inference.py

python ${NAROU_DIR}/scripts_for_kaggle/predict.py