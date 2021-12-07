NAROU_DIR=./nishika-narou-2021-1st-place-solution

python ${NAROU_DIR}/scripts/make_k_fold.py
python ${NAROU_DIR}/scripts/concat_pos_with_k_fold.py
python ${NAROU_DIR}/scripts/concat_features_with_posk_fold.py