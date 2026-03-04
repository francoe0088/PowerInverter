from utils_tools.data_preprocessing.main_deal import data_predict,loop_data_predict
import joblib
from utils_tools.model_storage.storage import load_last_model
import numpy as np 
import pandas as pd
#from utils_tools.model_inference.model_predict import predict_config
from utils_tools.data_preprocessing.load_yaml import predict_config
import json
import polars as pl
from utils_tools.data_preprocessing.main_deal import inverter_type_add
def read_columns():
    df_columns = pl.read_csv('./output/save_json/columns.csv')
    column_names = df_columns["Column Names"].to_list()
    print(len(column_names))
    return column_names

# 打印列名
#测试预测
def main_predict1(new_date):
    #column_names=read_columns()
    #print("clounm_names",column_names)
    #pl的格式
    test_file, seven_model_path, fifteen_model_path = predict_config()
    model_file = seven_model_path
    time_result_df=""
    error_sn=""
    stats_test,time_result_df,error_sn = loop_data_predict(test_file,new_date)
    #stats_test=pl.read_parquet("/home/franco/power/inverter1/inverter/code_v4/data/data_all/20250105_final_stats_test.parquet")
    stats_test=stats_test.to_pandas()
    stats_test=stats_test.fillna(0)
    #print('stats_test',stats_test)
    #sn_list=['TP40KBT000B240418015','TP25KBT000B240105233','TP25KBT000B240516192','TP30KBT000B240430189','TP20KBT000B240114035']
    #stats_test = stats_test.filter(pl.col('sn').is_in(sn_list))
    #print(stats_test)
    stats_test=stats_test
    #按照model的列进行排序？？？
    model=joblib.load('/home/franco/power/inverter1/inverter/code_v4/output/saved_models/7_30/zilin_model.joblib')
    #model = load_last_model()
    X_sn = stats_test['sn']
    X = stats_test.drop(['sn'],axis=1)  # 预测特征，排除 'sn' 和 'label'
    
    #X = X[column_names]
    #if not isinstance(X, pd.DataFrame):
        #X = pd.DataFrame(X, columns=column_names)
    # 获取模型的预测概率
    y_proba = model.predict_proba(X)
    
    # 打印出概率以便检查
    print(f"预测概率:\n{y_proba}")
    
    proba = 0.5# 二分类常用的概率阈值

    # 使用概率阈值进行二分类预测 (0.5 是常用的二分类阈值)
    y_pred = np.where(y_proba[:, 1] >= proba, 1, 0)
    
    
    # 打印预测结果
    print(f"预测结果:\n{y_pred}")

    # 如果预测结果全是1，考虑调整阈值
    if np.all(y_pred == 1):
        print("所有预测结果都是1，可能需要调整阈值或检查模型输出。")
    results = pd.DataFrame({'sn': X_sn, 'label': y_pred})
    results_label_1 = results[results['label'] == 1].reset_index(drop=True)
    results_label_0=results[results['label'] == 0].reset_index(drop=True)
    count_label_1 = results_label_1.shape[0]
    count_label_0=results_label_0.shape[0]
    #print(f"预测为1的样本数量: {count_label_1}")
    #print(f"预测为0的样本数量: {count_label_0}")
    
    return results_label_1,results_label_0,time_result_df,error_sn
#真实的预测
import numpy as np

def multi_threshold_predict(y_proba, thresh_0=0.3, thresh_1=0.7, thresh_2=0.7):
    """
    根据概率和阈值对样本进行多类别预测。
    
    参数:
    y_proba (list of list): 每个样本的概率 [p0, p1, p2]
    thresh_0 (float): 类别 0 的阈值 (可以不需要，但保留便于扩展)
    thresh_1 (float): 类别 1 的阈值
    thresh_2 (float): 类别 2 的阈值
    
    返回:
    np.ndarray: 每个样本的预测类别
    """
    y_pred = []
    for p0, p1, p2 in y_proba:
        if p2 >= thresh_2:
            y_pred.append(2)  # 类别 2
        elif p1 >= thresh_1:
            y_pred.append(1)  # 类别 1
        else:
            y_pred.append(0)  # 类别 0
    return np.array(y_pred)


    
def main_predict(new_date):
    test_file, seven_model_path, fifteen_model_path = predict_config()
    model_file = seven_model_path
    time_result_df=""
    error_sn=""
    stats_test,time_result_df,error_sn = loop_data_predict(test_file,new_date)
    #print("0000000000")
    #stats_test=pl.read_parquet("/home/franco/power/inverter1/inverter/code_v4/data/data_all/20250105_final_stats_test.parquet")
    stats_test=inverter_type_add(stats_test)
    #print("stats_test",stats_test)
    stats_test=stats_test.to_pandas()
    stats_test=stats_test.fillna(0)
    #print('stats_test',stats_test)
    #sn_list=['TP40KBT000B240418015','TP25KBT000B240105233','TP25KBT000B240516192','TP30KBT000B240430189','TP20KBT000B240114035']
    #stats_test = stats_test.filter(pl.col('sn').is_in(sn_list))
    #print(stats_test)
    stats_test=stats_test
    #按照model的列进行排序？？？
    #model=joblib.load('/home/franco/power/inverter1/inverter/code_v4/output/saved_models/7_30/zilin_model.joblib')
    model = load_last_model()
    model=model["model"]
    X_sn = stats_test['sn']
    X = stats_test.drop(['sn'],axis=1)  # 预测特征，排除 'sn' 和 'label'
    # 如果 X 不是 pandas DataFrame，则转换一下
    #if not isinstance(X, pd.DataFrame):
        #X = pd.DataFrame(X, columns=column_names)
    # 获取模型的预测概率
    y_proba = model.predict_proba(X)
    print("共多少",len(X))

    # 打印出概率以便检查
    #print(f"预测概率:\n{y_proba}")

    # 设置概率阈值
    #proba_threshold = 0.5
    y_pred = multi_threshold_predict(y_proba, thresh_0=0.26, thresh_1=0.80, thresh_2=0.80)
    # 预测 label：当概率 >= 阈值时 label=1，否则=0
    #y_pred = np.where(y_proba[:, 1] >= proba_threshold, 1, 0)

    # 如果预测结果全是1，考虑调整阈值
    if np.all(y_pred!=0):
        print("所有预测结果都是1，可能需要调整阈值或检查模型输出。")

    # 构造带预测结果和概率的表
    results = pd.DataFrame({
        'sn': X_sn,                  # 样本唯一标识或序列号
        'label': y_pred,             # 预测标签（0/1/2）
        'proba_label_0': y_proba[:, 0],
        'proba_label_1': y_proba[:, 1],
        'proba_label_2': y_proba[:, 2],
    })

    # 如果你只特别关心 1、2 这两个类别的概率，可以只保留后两列
    # results = pd.DataFrame({
    #     'sn': X_sn,
    #     'label': y_pred,
    #     'proba_label_1': y_proba[:, 1],
    #     'proba_label_2': y_proba[:, 2],
    # })

    # 分别查看预测为 0, 1, 2 的结果
    results_label_0 = results[results['label'] == 0].reset_index(drop=True)
    results_label_1 = results[results['label'] == 1].reset_index(drop=True)
    results_label_2 = results[results['label'] == 2].reset_index(drop=True)

    count_label_0 = results_label_0.shape[0]
    count_label_1 = results_label_1.shape[0]
    count_label_2 = results_label_2.shape[0]
    # 返回带概率的结果
    return results_label_0,results_label_1, results_label_2, time_result_df, error_sn




