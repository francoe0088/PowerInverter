'''
Author: franco
Date: 2024-11-26 15:01:46
LastEditors: franco
LastEditTime: 2024-11-27 14:45:05
Description: 
'''
from datetime import datetime
import joblib
import os
from utils_tools.model_storage.get_last_file import get_latest_file

def save_model(model, path):
    joblib.dump(model, path)

def load_last_model():
    path=get_latest_file()
    return joblib.load(path)

def model_save_path(model_name,pre_days2,need_days2):
    model_dir = f'./output/saved_models/{pre_days2}_{need_days2}'
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        #print(f"目录 {model_dir} 已创建")
    model_file = os.path.join(model_dir, f'{model_name}_{current_time}.joblib')
    return model_file
