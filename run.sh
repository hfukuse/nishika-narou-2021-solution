#Usage:
#sh ./nishika-narou-2021-1st-place-solution/run.sh
#

SCRIPT_DIR=./nishika-narou-2021-1st-place-solution/scripts

sh ${SCRIPT_DIR}/preprocess.sh
sh ${SCRIPT_DIR}/train.sh
sh ${SCRIPT_DIR}/predict.sh