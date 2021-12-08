SCRIPT_DIR=./nishika-narou-2021-1st-place-solution/scripts

#pretrain_modelを作成
python ${SCRIPT_DIR}/train/pretrain.py

#i8_modelを作成
python ${SCRIPT_DIR}/train/n8_nishika_narou_for_train.py
#i9_modelを作成
python ${SCRIPT_DIR}/train/n9_nishika_narou_for_train.py
#i18_modelを作成
python ${SCRIPT_DIR}/train/n18_nishika_narou_for_train.py
#i41_modelを作成
python ${SCRIPT_DIR}/train/n41_nishika_narou_for_train.py
#i42_modelを作成
python ${SCRIPT_DIR}/train/n42_nishika_narou_for_train.py
#i43_modelを作成
python ${SCRIPT_DIR}/train/n43_nishika_narou_for_train.py
