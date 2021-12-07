NAROU_DIR=./nishika-narou-2021-1st-place-solution
cd $NAROU_DIR

python ./scripts/make_k_fold.py
python ./scripts/concat_pos_with_k_fold.py
python ./scripts/concat_features_with_posk_fold.py