'''
Author: franco
Date: 2024-11-26 16:14:33
LastEditors: franco
LastEditTime: 2024-11-26 16:42:01
Description: 
'''
import yaml

def load_config(config_path='./configs/config.yaml'):
    """
    加载 YAML 配置文件并返回配置字典。

    参数:
        config_path (str): 配置文件的路径。

    返回:
        dict: 配置文件中的内容作为字典。
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)  # 读取文件内容并转换为字典
    return config

# 示例：加载配置文件并打印内容
def main_config():
    config_path='./configs/train_config.yaml'
    config = load_config('./configs/train_config.yaml')
    columns = config["columns"]
    model_name = config["model"]["name"]
    ok_data_path = config["data"]["ok"]["path"]
    ng_data_path = config["data"]["ng"]["path"]
    label1=config["labels"]["ok"]
    label2=config["labels"]["ng"]
    test_size=config["test_size"]
    tsfresh_filename_ok=config["tsfresh"]["ok_filename"]
    tsfresh_filename_ng=config["tsfresh"]["ng_filename"]
    pre_days_ok=config["pre_days"]["ok"]
    pre_days_ng=config["pre_days"]["ng"]
    need_days_ok=config["need_days"]["ok"]
    need_days_ng=config["need_days"]["ng"]
    return columns,model_name,ok_data_path,ng_data_path,label1,label2,test_size,tsfresh_filename_ok,tsfresh_filename_ng,pre_days_ok,pre_days_ng,need_days_ok,need_days_ng

def predict_config():
    config_path='./configs/predict_config.yaml'
    config = load_config(config_path)
    test_file=config["data"]["path"]
    seven_model_path=config["model"]["path"]["seven_day"]
    fifiteen_model_path=config["model"]["path"]["fifiteen_day"]
    return test_file,seven_model_path,fifiteen_model_path