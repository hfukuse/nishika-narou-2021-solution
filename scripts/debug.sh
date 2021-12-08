SCRIPT_DIR=./nishika-narou-2021-1st-place-solution/scripts

#pretrain_modelを作成
python ${SCRIPT_DIR}/train/pretrain.py --debug

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
#n117_modelを作成
python ${SCRIPT_DIR}/train/n117_nishika_narou_for_train.py --debug