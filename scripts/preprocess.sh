SCRIPT_DIR=./nishika-narou-2021-1st-place-solution/scripts

python ${SCRIPT_DIR}/preprocess/make_k_fold.py
python ${SCRIPT_DIR}/preprocess/concat_pos_with_k_fold.py
python ${SCRIPT_DIR}/preprocess/concat_features_with_posk_fold.py