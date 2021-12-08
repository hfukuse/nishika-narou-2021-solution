#Usage:
#sh ./nishika-narou-2021-1st-place-solution/run_inference_using_trained_models.sh
#
#事前に学習済みmodelを./nishika-narou-2021-1st-place-solution/modelsに設置しておいてください
#
SCRIPT_DIR=./nishika-narou-2021-1st-place-solution/scripts

sh ${SCRIPT_DIR}/preprocess.sh

#bertによる予測
sh ${SCRIPT_DIR}/predict_bert.py

#catboostによる予測
sh ${SCRIPT_DIR}/predict_catboost.py

#アンサンブルするmodel1による予測
sh ${SCRIPT_DIR}/predict_model1.py

#アンサンブルするmodel2による予測
sh ${SCRIPT_DIR}/predict_model2.py