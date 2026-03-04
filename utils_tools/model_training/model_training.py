
'''
Author: franco
Date: 2024-11-26 14:57:42
LastEditors: franco
LastEditTime: 2024-11-27 14:38:49
Description: 
'''
from utils_tools.model_training.model_definition import ModelEvaluator
from utils_tools.data_preprocessing.main_deal import data_split1
from utils_tools.data_preprocessing.main_deal import data_split
from utils_tools.data_preprocessing.reader import read_file
from utils_tools.data_preprocessing.feature_engineering import featureProcessor
import yaml
import joblib
from utils_tools.model_storage.storage import model_save_path
from utils_tools.model_interpretation.interpretation import Confusion
from utils_tools.data_preprocessing.load_yaml import main_config

def train(ok_filename,ng_filename,label1,label2,pre_days1,need_days1,pre_days2,need_days2,tsfresh_filename_ok,tsfresh_filename_ng,model_name,test_size):
    X_train, X_test, y_train, y_test=data_split(ok_filename,ng_filename,label1,label2,pre_days1,need_days1,pre_days2,need_days2,tsfresh_filename_ok,tsfresh_filename_ng,test_size)
    evaluator = ModelEvaluator()
    model,y_pred = evaluator.evaluate_model(model_name,X_train, X_test, y_train, y_test)
    model_file=model_save_path(model_name,pre_days2,need_days2)
    #print(model_file)
    joblib.dump(model,model_file)
    Confusion(y_test, y_pred,need_days2,"test")
    
    return X_train, y_train,X_test,y_test,y_pred

def main():
    columns,model_name,ok_filename,ng_filename,label1,label2,test_size,tsfresh_filename_ok,tsfresh_filename_ng,pre_days_ok,pre_days_ng,need_days_ok,need_days_ng=main_config()
    X_train, y_train,X_test,y_test,y_pred=train(ok_filename,ng_filename,label1,label2,pre_days_ok,need_days_ok,pre_days_ng,need_days_ng,tsfresh_filename_ok,tsfresh_filename_ng,model_name,test_size)
#main_train()
    


    

    

