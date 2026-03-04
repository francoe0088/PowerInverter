'''
Author: franco
Date: 2024-11-26 15:28:13
LastEditors: franco
LastEditTime: 2024-11-27 15:36:27
Description: 
'''
from tqdm import tqdm

import pandas as pd
from utils_tools.data_preprocessing.reader import read_file
import tqdm
from utils_tools.data_preprocessing.data_deal import data_combined,get_train_test_split
from utils_tools.data_preprocessing.feature_engineering import featureProcessor
#from utils_tools.data_preprocessing.polar_processing import featureProcessor
import numpy as np
import time
import yaml
import polars as pl
from utils_tools.data_preprocessing.load_predict_columns import load_columns
import os
from datetime import datetime
f = open(r'./configs/train_config.yaml', encoding='utf-8')  # config file
hyp = yaml.load(f, Loader=yaml.FullLoader)


def data_split1(ok_filename,ng_filename,label1,label2,pre_days1,need_days1,pre_days2,need_days2,tsfresh_filename_ok,tsfresh_filename_ng,test_size):
    print("0.0.00")
    all_columns=None
    df_ok=read_file(ok_filename)[hyp["columns"]]
    ##print("00000000000000000000000000000000")
    df_ng=read_file(ng_filename)[hyp["columns"]]
    #print('00000')
    processor_ng=featureProcessor(df_ng,all_columns)
    #print("========")
    stats_ng,all_columns=processor_ng.compute_features_with(pre_days2,need_days2,tsfresh_filename_ng)
    #print("000000000")
    processor_ok=featureProcessor(df_ok,all_columns)
    stats_ok,all_columns=processor_ok.compute_features_with(pre_days1,need_days1,tsfresh_filename_ok)
    #stats_ng=pl.from_pandas(stats_ng)
    #stats_ok=pl.from_pandas(stats_ok)
    #stats_ok=pl.read_parquet('/home/franco/power/inverter1/inverter/code_v4/data/data_all/20250104_final_stats_test.parquet')
    #all_df=pl.read_parquet('./data/train/processed/dataset_ng/train_ng_all_sampled.parquet')
    #errror=pl.read_parquet('/home/franco/power/inverter1/inverter/code_v4/data/train/processed/dataset_ng/ng_data.parquet')
    #stats_ng = stats_ng.filter(~pl.col('sn').is_in(errror['sn']))

    #stats_ng=pl.from_pandas(stats_ng)
    #stats_ok=pl.from_pandas(stats_ok)
    
    #num_samples = 280 if len(stats_ok) >= 280 else len(stats_ok)
    #stats_ok = stats_ok.sample(n=num_samples, with_replacement=False)
    common_columns = list(set(stats_ok.columns) & set(stats_ng.columns))

    # 只保留相同的列
    stats_ok = stats_ok.select(*[pl.col(col) for col in common_columns] + [pl.lit(label1).alias('label')])
    stats_ng = stats_ng.select(*[pl.col(col) for col in common_columns] + [pl.lit(label2).alias('label')])

    #stats_ok.to_parquet("ok.parquet")
    #print("stats_ng",stats_ng.shape)
    #print("stats_ok",stats_ok.shape)
    combined_monthly_df=data_combined(stats_ok,stats_ng)
    X_train, X_test, y_train, y_test=get_train_test_split(combined_monthly_df,test_size)


    print(f"模型已存saved_models文件夹中,其中ok数据{len(stats_ok)}条,NG数据{len(stats_ng)}条")
    return X_train, X_test, y_train, y_test
import polars as pl

def train_one_Seven(
    df_ok,
    df_ng,
    label1,
    label2,
    pre_days1,
    need_days1,
    pre_days2,
    need_days2,
    tsfresh_filename_ok,
    tsfresh_filename_ng,
    all_columns
):
    # 计算 NG 特征
    processor_ng = featureProcessor(df_ng, all_columns)
    stats_ng, all_columns = processor_ng.compute_features_with(pre_days2, need_days2, tsfresh_filename_ng)
    print("ng已完成")


    common_columns = []

    if pre_days2 < 2:
        # 计算 OK 特征
        processor_ok = featureProcessor(df_ok, all_columns)
        stats_ok, all_columns = processor_ok.compute_features_with(pre_days1, need_days1, tsfresh_filename_ok)
        #stats_ok=pl.read_parquet('/home/franco/power/inverter1/inverter/code_v4/data/data_all/20250104_final_stats_test.parquet')

        print("======")
        # 对 OK 和 NG 的列取交集
        common_columns = list(set(stats_ok.columns) & set(stats_ng.columns))
        print(common_columns)

        # 只保留共同列 + label 列
        stats_ok = stats_ok.select(
            *[pl.col(col) for col in common_columns],  # OK
            pl.lit(label1).alias('label')
        )
    else:
        pass
        # 这里不计算 OK，所以 stats_ok = None
        

    # 如果没在 if 里对 common_columns 赋值，说明 pre_days2>=2
    # 那么 NG 的列算是“通用列”了
    if not common_columns:
        common_columns = stats_ng.columns

    # NG 也只保留跟 OK 相同的列 + label
    stats_ng = stats_ng.select(
        *[pl.col(col) for col in common_columns],
        pl.lit(label2).alias('label')
    )
    if pre_days2 >1:
        schema = stats_ng.schema

        # 创建一个与 ng 具有相同列类型的空表格
        stats_ok = pl.DataFrame(schema={col: dtype for col, dtype in schema.items()})
        print(stats_ok)
        #print(type)

    #print("0.0.0.0.0")
    #print(stats_ng['label'])
    # 统一用 data_combined 来做合并，以保证输出的列是一致的
    print(type(stats_ok),type(stats_ng)) 
    error_type=pl.read_csv("/home/franco/power/inverter1/inverter/code_v6/filter_sn/error_type/error_type.csv")
    stats_ok = stats_ok.with_columns(pl.lit(0).alias("error_type"))
    stats_ng = stats_ng.join(error_type, on="sn", how="left")
    stats_ng = stats_ng.filter(pl.col("error_type").is_not_null())

    # 去掉原 label 列(如果存在)
    #stats_ng = stats_ng.drop("label")

    # 将 error_type 列改名为 label
    #stats_ng = stats_ng.rename({"error_type": "label"})
    combined_monthly_df = data_combined(stats_ok, stats_ng)

    #print('ok',stats_ok['sn'])
    #print('ng',stats_ng['sn'])
    #print('1111118888888',combined_monthly_df['sn'])
    #print("11111111111111111")
    return combined_monthly_df, all_columns


def data_split(ok_filename,ng_filename,label1,label2,pre_days1,need_days1,pre_days2,need_days2,tsfresh_filename_ok,tsfresh_filename_ng,test_size):
    all_columns = None
    combined_all = None
    all_columns=None
    df_ok=read_file(ok_filename)[hyp["columns"]]
    ##print("00000000000000000000000000000000")
    df_ng=read_file(ng_filename)[hyp["columns"]]

    
    pre_days2_list = [0,1, 2, 3, 4, 5, 6, 7]
    #pre_days2_list = [1, 2]
    
    for pre_days2 in pre_days2_list:
        #print("0909049i039")
        combined_monthly_df, all_columns = train_one_Seven(
            df_ok, df_ng, label1, label2, pre_days1, need_days1, int(pre_days2), need_days2, 
            tsfresh_filename_ok, tsfresh_filename_ng, all_columns
        )
        #print("0909049i039")
        
        if combined_all is None:
            combined_all = combined_monthly_df
        else:
            combined_monthly_df = combined_monthly_df.select(combined_all.columns)
            combined_all = pl.concat([combined_all, combined_monthly_df], how="vertical")
    #print(combined_all['sn'])

    combined_all=inverter_type_add(combined_all)
    #data_np = combined_all.to_numpy()
    combined_all = combined_all.sample(n=combined_all.height, with_replacement=False, shuffle=True)
        #stats_ng = stats_ng.drop("label")

    # 将 error_type 列改名为 label
    #stats_ng = stats_ng.rename({"error_type": "label"})
    combined_all = combined_all.drop("label")
    combined_all=combined_all.rename({"error_type": "label"})
    combined_all.write_csv('combined_all.csv')
    X_train, X_test, y_train, y_test=get_train_test_split(combined_all,test_size)
    # 根据 test_size 进行数据集拆分（可选）
    return X_train, X_test, y_train, y_test

# 示例调用
# 确保 hyp 已定义，并且包含 "columns" 键
#加逆变器型号特征Inverter_type





import polars as pl

def inverter_type_add(combined_all) -> pl.DataFrame:
    # 定义“字符串 -> float”的映射
    #inverter_type_mapping = {
    #    "TP10KB": 0.0,
    #    "TP12KB": 1.0,
    #    "TP15KB": 2.0,
    #    "TP17KB": 3.0,
    #    "TP20KB": 4.0,
    #    "TP25KB": 5.0,
    #    "TP30KB": 6.0,
    #    "TP33KB": 7.0,
    #    "TP36KB": 8.0,
    #    "TP40KB": 9.0
    #}

    inverter_type_mapping={    
        "TP10KB": 0.0,
        "TP12KB": 1.0,
        "TP15KB": 2.0,
        "TP17KB": 3.0,
        "TP20KB": 4.0,
        "TP25KB": 5.0,
        "TP30KB": 6.0,
        "TP33KB": 7.0,
        "TP36KB": 8.0,
        "TP40KB": 9.0
    }

    # 1) 先把 sn 列转成字符串（以便做 .str.slice(0, 6)）
    #    同时保留一份原始的 sn 列，以备填充 map_dict 失败的场景
    combined_all = combined_all.with_columns([
        pl.col('sn').cast(pl.Utf8).alias('sn_str')  
    ])
    print('sn',combined_all['sn'])
    # 2) 对 sn_str 截取前6位，做字典映射
    # 3) 如果 map_dict 结果是 null（说明无法匹配），就用原始的 sn 填充
    # 4) 再整体转换成 float64
    combined_all = combined_all.with_columns([
        (
            pl.col('sn_str')
            .str.slice(0, 6)
            .map_dict(inverter_type_mapping)        # dict: "TP10KB" -> 0.0, ...
            .fill_null(pl.col('sn'))               # 如果没匹配，则用原 sn (float64)
            .cast(pl.Float64)                      # 最后转成 float64
        ).alias('inverter_type_list')
    ])

    # 5) 根据需求删除临时列 sn_str
    combined_all = combined_all.drop('sn_str')
    #print(combined_all['inverter_type_list'])
    return combined_all







def data_split1(ok_filename,ng_filename,label1,label2,pre_days1,need_days1,pre_days2,need_days2,tsfresh_filename_ok,tsfresh_filename_ng,test_size):
    stats_ok=read_file(r'./data/tsfresh_feature/20250106_0457_train2_2_9_feature_ok_7_day.parquet')
    stats_ng=read_file(r'./data/tsfresh_feature/20250106_0445_train2_2_9_feature_ng_7_day.parquet')
    stats_ng1=read_file(r'./data/tsfresh_feature/20250106_0441_train2_2_9_feature_ng_7_day.parquet')
    stats_ng1=stats_ng1.head(4)
    print(stats_ng1)
    # 使用 slice 从第 4 行开始提取，注意索引从 0 开始
    remaining_data = stats_ng1.slice(4)

# 6. 保存剩余的数据到本地
    remaining_output_path = r'./data/tsfresh_feature/remaining_data_ng.parquet'
    remaining_data.write_parquet(remaining_output_path)
    
    common_columns = list(set(stats_ok.columns) & set(stats_ng.columns)&set(stats_ng1.columns))
    stats_ok = stats_ok.select(*[pl.col(col) for col in common_columns] + [pl.lit(label1).alias('label')])
    stats_ng = stats_ng.select(*[pl.col(col) for col in common_columns] + [pl.lit(label2).alias('label')])
    stats_ng1 = stats_ng1.select(*[pl.col(col) for col in common_columns] + [pl.lit(label2).alias('label')])
    X_six = stats_ng1.drop(['label','sn'])  # 特征
    y_six = stats_ng1['label']  # 标签

    #stats_ok=stats_ok.with_columns(pl.lit(0).alias('label'))
    #stats_ng=stats_ng.with_columns(pl.lit(1).alias('label'))
    #stats_ng1=stats_ng1.with_columns(pl.lit(1).alias('label'))
    combined_monthly_df=data_combined(stats_ok,stats_ng)
    X_train, X_test, y_train, y_test=get_train_test_split(combined_monthly_df,test_size)
    X_six = stats_ng1.drop(['label','sn'])  # 特征
    y_six = stats_ng1['label']  # 标签
    X_train=data_combined(X_train,X_six)

    y_train = y_train.append(y_six)
    #y_train=data_combined(y_train,y_six)

    
    return X_train, X_test, y_train, y_test

def data_predict(test_file):
    all_columns=None
    all_columns=load_columns("./output/columns_vpv_ipv/columns_vpv_ipv.pkl")
     # 文件夹路径
    #test_files = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]
    df_test=read_file(test_file)[hyp["columns"]]
    processor_test=featureProcessor(df_test,all_columns)
    tsfresh_test="test.parquet"
    stats_test,all_columns=processor_test.compute_features_with_predict(0,30,tsfresh_test)
    ##print("stats_test:",stats_test.shape)
    ##print("stats_ok",stats_ok)
    #stats_test=pl.read_parquet('./data/tsfresh_feature/20241205_0248_train2_2_9_feature_ok_7_day.parquet')
    stats_test=pl.from_pandas(stats_test)
    return stats_test



from tqdm import tqdm
def loop_data_predict(test_folder, new_date):
    new_date1=new_date
    # 加载已知的列顺序
    all_columns = load_columns("./output/columns_vpv_ipv/columns_vpv_ipv.pkl")

    test_files = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]
    
    all_stats_test = []
    base_columns = None
    new_date = str(new_date) 
    new_date = datetime.strptime(new_date + ' 00:00:00', '%Y%m%d %H:%M:%S')
    all_sn_list = []
    all_time_list = []
    other_error=[]
    
    for idx, test_file in enumerate(tqdm(test_files, desc="Processing files")):
        file_path = os.path.join(test_folder, test_file)
        df_test = read_file(file_path)[hyp["columns"]]

        # 处理特征
        processor_test = featureProcessor(df_test, all_columns)
        tsfresh_test = "test.parquet"
        stats_test, _, sn_list,time_list,error_sn = processor_test.compute_features_with_predict(0, 7, tsfresh_test, new_date)
        
        # 转换为 Polars DataFrame
        #stats_test = pl.from_pandas(stats_test)

        # 如果是第一个文件，则记录下列顺序
        if base_columns is None:
            base_columns = stats_test.columns

        # 对 stats_test 进行列排序（根据首个文件的列顺序）
        stats_test = stats_test.select(base_columns)

        all_stats_test.append(stats_test)
        all_sn_list.append(sn_list)
        all_time_list.append(time_list)
        other_error.append(error_sn)

        time.sleep(0)  # 可选，根据需要保留或移除
    
    # 合并所有 DataFrame
        
    final_stats_test = pl.concat(all_stats_test, how="vertical")
    #final_stats_test.write_parquet(f"./data/data_all/{new_date1}_final_stats_test.parquet")
    
    # 合并 SN 和 Time 列表
    final_all_sn_list = [item for sublist in all_sn_list for item in sublist]
    final_all_time_list = [item for sublist in all_time_list for item in sublist]
    final_all_other_error_list=[item for sublist in other_error for item in sublist]


    # 创建一个 Polars DataFrame
    result_df = pl.DataFrame({
        "sn": final_all_sn_list,
        "time": final_all_time_list
    })
    # 保存为 CSV 文件
    #result_df.write_csv(f"./data/data_sn/sn_result_data_{new_date1}.csv")


    # 返回最终的 DataFrame
    return final_stats_test, result_df,other_error







