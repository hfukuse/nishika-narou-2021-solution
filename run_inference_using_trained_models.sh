#Usage:
#sh ./nishika-narou-2021-solution/run_inference_using_trained_models.sh
#
#事前に学習済みmodelを./nishika-narou-2021-solution/modelsに設置しておいてください
#
SCRIPT_DIR=./nishika-narou-2021-solution/scripts

sh ${SCRIPT_DIR}/preprocess.sh

#bertによる予測
sh ${SCRIPT_DIR}/predict_bert.sh

#catboostによる予測
sh ${SCRIPT_DIR}/predict_catboost.sh

#アンサンブルするmodel1による予測
sh ${SCRIPT_DIR}/predict_model1.sh

#アンサンブルするmodel2による予測
sh ${SCRIPT_DIR}/predict_model2.sh