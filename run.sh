#Usage:
#sh ./nishika-narou-2021-solution/run.sh
#

SCRIPT_DIR=./nishika-narou-2021-solution/scripts

sh ${SCRIPT_DIR}/preprocess.sh
sh ${SCRIPT_DIR}/train_bert.sh
sh ${SCRIPT_DIR}/predict_bert.sh
sh ${SCRIPT_DIR}/train_catboost.sh
sh ${SCRIPT_DIR}/predict_catboost.sh
sh ${SCRIPT_DIR}/train_model1.sh
sh ${SCRIPT_DIR}/predict_model1.sh
sh ${SCRIPT_DIR}/train_model2.sh
sh ${SCRIPT_DIR}/predict_model2.sh
