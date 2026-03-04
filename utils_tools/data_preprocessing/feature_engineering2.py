# src/feature_engineering/polar_processing.py
import polars as pl
import yaml
import os
from functools import reduce, wraps
from datetime import datetime, timedelta
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
import time
import pandas as pd
import functime
import numpy as np
#from utils_tools.utilities.wraps_fun import log_execution_time
from functime.feature_extractors import fft_coefficients, autoregressive_coefficients, number_cwt_peaks
from utils_tools.data_preprocessing.data_deal import DataProcessor
def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        #print(f"执行函数 {func.__name__}...")
        result = func(*args, **kwargs)
        end_time = time.time()
        #print(f"函数 {func.__name__} 执行完成，耗时 {end_time - start_time:.2f} 秒")
        return result
    return wrapper
def fft(x):
    result = fft_coefficients(x)
    return {'real_1': result['real'][0], 'real_2':result['real'][1],
            'imag_1': result['imag'][0], 'imag_2':result['imag'][1],}
class partial_auto:
    def __init__(self, n_lags):
        self.n_lags = n_lags
    
    def add(self, x):
        try:
            return autoregressive_coefficients(x, self.n_lags)[0]
        except:
            return 0.0
class add_prefix_to_struct_keys:
    def __init__(self, prefix):
        self.prefix = prefix
    
    def add(self, struct_dict):
        new_dict = {}
        for key, value in struct_dict.items():
            new_key = self.prefix + key
            new_dict[new_key] = value
        return new_dict
def split_structs(df,numeric_cols):
    col_list = []
    for col in numeric_cols:
        if col != 'label' :
            df = df.with_columns(
                pl.col(f'{col}{col}').map_elements(add_prefix_to_struct_keys(f'{col}_').add),
            )
            df = df.with_columns(
                pl.col(f'{col}_fft').map_elements(add_prefix_to_struct_keys(f'{col}_').add),
            )
            col_list.append(f'{col}{col}')
            col_list.append(f'{col}_fft')
    df = df.unnest(col_list)
    for col in df.columns:
        if col != 'sn' and (df[col].is_infinite()).any():
            col_max = df.filter(pl.col(col)!= np.inf)[col].max()
            df = df.with_columns(pl.when(pl.col(col) == np.inf)
                                 .then(col_max).otherwise(pl.col(col)).alias(col)
            )
    return df
class featureProcessor:
    @log_execution_time
    def __init__(self, df: pl.DataFrame,all_columns):
        #self.df = df
        #添加数据处理模块
        self.df,self.all_columns,self.sn_na_list=DataProcessor(df,all_columns).main()

    #@log_execution_time
    @log_execution_time
    def compute_features_with_tsfresh(self, days, outdays, tsfresh_filename):
        import time
        time1=time.time()
        # 确保 'createtime' 是 datetime 类型
        
        self.df = self.df.with_columns(
            pl.col('createtime').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
        )

        # 重置索引 (Polars 不支持直接重置索引，但可以通过添加行号)
        df_reset = self.df.with_row_count("index")

        # 确保 'sn' 被视为字符串类型
        if 'sn' in df_reset.columns:
            df_reset = df_reset.with_columns(pl.col('sn').cast(pl.Utf8))
        else:
            #print("Error: 'sn' 列不存在于 DataFrame 中。")
            return pl.DataFrame()

        # 选择数值列（排除 'sn' 列）
        numeric_cols = [
        col for col, dtype in zip(df_reset.columns, df_reset.dtypes)
        if dtype in pl.NUMERIC_DTYPES and col not in ('sn', 'createtime')]
        ##print("numeric_cols",numeric_cols)
        # 初始化用于收集结果的列表
        feature_data = []

        # 定义自定义特征提取参数
        custom_fc_parameters = {
            "mean": None, "median": None, "minimum": None, "maximum": None,
            "standard_deviation": None, "variance": None, "skewness": None, "kurtosis": None,
            "first_location_of_maximum": None, "first_location_of_minimum": None,
            "absolute_sum_of_changes": None, "count_above_mean": None, "count_below_mean": None,
            "autocorrelation": [{"lag": lag} for lag in range(1, 4)],
            "partial_autocorrelation": [{"lag": lag} for lag in range(1, 4)],
            "fft_coefficient": [
                {"coeff": 1, "attr": "real"}, {"coeff": 1, "attr": "imag"},
                {"coeff": 2, "attr": "real"}, {"coeff": 2, "attr": "imag"}
            ],
            "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in range(1, 3)],
            "number_peaks": [{"n": 1}, {"n": 3}],
            "number_cwt_peaks": [{"n": 1}, {"n": 3}],
            "binned_entropy": [{"max_bins": 10}],
            "linear_trend": [{"attr": attr} for attr in ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']],
            "quantile": [{"q": q} for q in [0.1, 0.25, 0.5, 0.75, 0.9]],
            "range_count": [{"min": -1, "max": 1}],
        }

        import time
        start_time1 = time.time()

        # 使用 partition_by 来迭代每个 'sn' 组
        groups = df_reset.partition_by("sn")

        for group_df in groups:
            # 获取当前组的 'sn' 值
            
            sn = group_df["sn"][0]

            # 获取组内的最大 createtime
            new_time_series = group_df.select(pl.col('createtime').max()).to_dict(as_series=False)
            if 'createtime' not in new_time_series or not new_time_series['createtime']:
                continue  # 如果没有 createtime，跳过

            new_time = new_time_series['createtime'][0]
            if isinstance(new_time, datetime):
                # already a datetime object
                pass
            else:
                # 如果 new_time 不是 datetime 对象，根据实际情况进行转换
                # 这里假设 new_time 是一个字符串，您可以根据需要调整
                new_time = datetime.strptime(new_time, "%Y-%m-%d %H:%M:%S")

            # 计算时间窗口
            end_time = new_time - timedelta(days=days)
            start_time_window = end_time - timedelta(days=outdays)

            # 在时间窗口内过滤数据
            window_data = group_df.filter(
                (pl.col('createtime') >= start_time_window) & (pl.col('createtime') < end_time)
            )

            if not window_data.is_empty():
                feature_data.append(window_data)
        end_time1 = time.time()
        ##print(f"消耗时间：{end_time1 - start_time1} 秒")

        # 检查是否有可用的特征数据
        if len(feature_data) == 0:
            #print("没有可用的数据在指定的时间窗口内。")
            return pl.DataFrame()  # 如果没有数据，返回空的 DataFrame

        # 合并所有特征数据
        try:
            all_data = pl.concat(feature_data,how="diagonal")
        except Exception as e:
            #print("Error during concatenating feature data:", e)
            return pl.DataFrame()

        ##print("合并后的数据预览：")
        ##print(all_data.head())

        # 确保 'sn' 和 'createtime' 存在于 all_data 中
        #if not {'sn', 'createtime'}.issubset(all_data.columns):
        #    #print("Error: 'sn' 或 'createtime' 列在合并后的数据中缺失。")
        #    return pl.DataFrame()

        # 使用 tsfresh 需要的数据格式，将数据从宽格式转换为长格式
        try:
            melted = all_data.melt(
                id_vars=['sn', 'createtime'],
                value_vars=numeric_cols,
                variable_name='kind',
                value_name='value'
            )
        except Exception as e:
            #print("Error during melting the dataframe:", e)
            return pl.DataFrame()
        time2=time.time()
        ##print(f"花费了{time2-time1}时间！")
        ##print("三小")
        ##print("转换后的长格式数据预览：")
        ##print(melted.head())

        # 使用自定义参数提取特征
        try:
            ##print("三小")
            extracted_features = extract_features(
                melted.to_pandas(),  # tsfresh 目前不支持 polars，因此需要转换为 pandas
                column_id='sn',
                column_sort='createtime',
                column_kind='kind',
                column_value='value',
                default_fc_parameters=custom_fc_parameters,
                disable_progressbar=True
                #n_jobs=-1
            )
            ##print("结束")
        except Exception as e:
            #print("Error during feature extraction:", e)
            return pl.DataFrame()

        ##print("提取的特征预览：")
        ##print(extracted_features.head())

        # 填补缺失值
        try:
            ##print("传回数据啦")
            features_df = impute(extracted_features).reset_index()
        except Exception as e:
            #print("Error during imputing missing values:", e)
            return pl.DataFrame()

        # 重命名列
        features_df = features_df.rename(columns={'index': 'sn'})
        features_df = features_df.loc[:, ~features_df.columns.str.contains('index')]

        ##print("填补缺失值后的特征数据预览：")
        ##print(features_df.head())

        # 将特征保存为 Parquet 文件
        try:
            path = "./data/tsfresh_feature"
            os.makedirs(path, exist_ok=True)
            current_date = datetime.now().strftime('%Y%m%d_%H%M')
            parquet_path = os.path.join(path, f'{current_date}_{tsfresh_filename}')
            features_df.to_parquet(parquet_path)
            print(f"特征数据已保存为 {parquet_path}")
        except Exception as e:
            #print("Error during saving to Parquet:", e)
            return pl.DataFrame()

        # 将 pandas DataFrame 转换为 polars DataFrame 后返回
        try:
            #return pl.from_pandas(features_df)
            #print("传回数据啦")
            return features_df,self.all_columns
        except Exception as e:
            #print("Error during converting pandas DataFrame to polars DataFrame:", e)
            return pl.DataFrame()
        
########################################
    def compute_features_with_tsfresh_predict_old(self, days, outdays, tsfresh_filename,new_time):
        import time
        time1=time.time()
        # 确保 'createtime' 是 datetime 类型
        
        self.df = self.df.with_columns(
            pl.col('createtime').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
        )


        # 重置索引 (Polars 不支持直接重置索引，但可以通过添加行号)
        df_reset = self.df.with_row_count("index")

        # 确保 'sn' 被视为字符串类型
        if 'sn' in df_reset.columns:
            df_reset = df_reset.with_columns(pl.col('sn').cast(pl.Utf8))
        else:
            #print("Error: 'sn' 列不存在于 DataFrame 中。")
            return pl.DataFrame()

        # 选择数值列（排除 'sn' 列）
        numeric_cols = [
        col for col, dtype in zip(df_reset.columns, df_reset.dtypes)
        if dtype in pl.NUMERIC_DTYPES and col not in ('sn', 'createtime')]
        #print("numeric_cols",numeric_cols)
        # 初始化用于收集结果的列表
        feature_data = []

        # 定义自定义特征提取参数
        custom_fc_parameters = {
            "mean": None, "median": None, "minimum": None, "maximum": None,
            "standard_deviation": None, "variance": None, "skewness": None, "kurtosis": None,
            "first_location_of_maximum": None, "first_location_of_minimum": None,
            "absolute_sum_of_changes": None, "count_above_mean": None, "count_below_mean": None,
            "autocorrelation": [{"lag": lag} for lag in range(1, 4)],
            "partial_autocorrelation": [{"lag": lag} for lag in range(1, 4)],
            "fft_coefficient": [
                {"coeff": 1, "attr": "real"}, {"coeff": 1, "attr": "imag"},
                {"coeff": 2, "attr": "real"}, {"coeff": 2, "attr": "imag"}
            ],
            "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in range(1, 3)],
            "number_peaks": [{"n": 1}, {"n": 3}],
            "number_cwt_peaks": [{"n": 1}, {"n": 3}],
            "binned_entropy": [{"max_bins": 10}],
            "linear_trend": [{"attr": attr} for attr in ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']],
            "quantile": [{"q": q} for q in [0.1, 0.25, 0.5, 0.75, 0.9]],
            "range_count": [{"min": -1, "max": 1}],
        }

        import time
        start_time1 = time.time()
        skipped_sns = [] 
        # 使用 partition_by 来迭代每个 'sn' 组
        groups = df_reset.partition_by("sn")
        end_time = new_time - timedelta(days=days)
        start_time_window = end_time - timedelta(days=outdays)
        print(start_time_window,end_time)

        for group_df in groups:
            #print(len(group_df))
            # 获取当前组的 'sn' 值
            sn = group_df["sn"][0]

            # 获取组内的最大 createtime
            #snew_time_series = group_df.select(pl.col('createtime').max()).to_dict(as_series=False)
            #sif 'createtime' not in new_time_series or not new_time_series['createtime']:
            #s    continue  # 如果没有 createtime，跳过

            #new_time = new_time_series['createtime'][0]
            #if isinstance(new_time, datetime):
                # already a datetime object
                #pass
            #else:
                # 如果 new_time 不是 datetime 对象，根据实际情况进行转换
                # 这里假设 new_time 是一个字符串，您可以根据需要调整
            

            # 计算时间窗口

            # 在时间窗口内过滤数据
            window_data = group_df.filter(
                (pl.col('createtime') >= start_time_window) & (pl.col('createtime') < end_time)
            )

            if not window_data.is_empty():
                if len(window_data)<2000:
                    skipped_sns.append(sn)
                    continue
                feature_data.append(window_data)
            #print('window_data',window_data)
        df_skipped_sns = pl.DataFrame({"sn": skipped_sns})
        # 保存跳过的 sn 值到 CSV 文件
        df_skipped_sns.write_csv(f'{str(new_time)}_skipped_sns.csv')
        end_time1 = time.time()
        ##print(f"消耗时间：{end_time1 - start_time1} 秒")

        # 检查是否有可用的特征数据
        if len(feature_data) == 0:
            #print("没有可用的数据在指定的时间窗口内。")
            return pl.DataFrame()  # 如果没有数据，返回空的 DataFrame

        # 合并所有特征数据
        try:
            all_data = pl.concat(feature_data,how="diagonal")
        except Exception as e:
            #print("Error during concatenating feature data:", e)
            return pl.DataFrame()

        ##print("合并后的数据预览：")
        ##print(all_data.head())

        # 确保 'sn' 和 'createtime' 存在于 all_data 中
        #if not {'sn', 'createtime'}.issubset(all_data.columns):
        #    #print("Error: 'sn' 或 'createtime' 列在合并后的数据中缺失。")
        #    return pl.DataFrame()

        # 使用 tsfresh 需要的数据格式，将数据从宽格式转换为长格式
        try:
            melted = all_data.melt(
                id_vars=['sn', 'createtime'],
                value_vars=numeric_cols,
                variable_name='kind',
                value_name='value'
            )
        except Exception as e:
            #print("Error during melting the dataframe:", e)
            return pl.DataFrame()
        time2=time.time()
        ##print(f"花费了{time2-time1}时间！")
        ##print("三小")
        ##print("转换后的长格式数据预览：")
        ##print(melted.head())

        # 使用自定义参数提取特征
        try:
            ##print("三小")
            extracted_features = extract_features(
                melted.to_pandas(),  # tsfresh 目前不支持 polars，因此需要转换为 pandas
                column_id='sn',
                column_sort='createtime',
                column_kind='kind',
                column_value='value',
                default_fc_parameters=custom_fc_parameters,
                disable_progressbar=True
                #n_jobs=-1
            )
            ##print("结束")
        except Exception as e:
            print("Error during feature extraction:", e)
            return pl.DataFrame()

        ##print("提取的特征预览：")
        ##print(extracted_features.head())

        # 填补缺失值
        try:
            ##print("传回数据啦")
            features_df = impute(extracted_features).reset_index()
        except Exception as e:
            print("Error during imputing missing values:", e)
            return pl.DataFrame()

        # 重命名列
        features_df = features_df.rename(columns={'index': 'sn'})
        features_df = features_df.loc[:, ~features_df.columns.str.contains('index')]

        ##print("填补缺失值后的特征数据预览：")
        ##print(features_df.head())

        # 将特征保存为 Parquet 文件
        try:
            path = "./data/tsfresh_feature"
            os.makedirs(path, exist_ok=True)
            current_date = datetime.now().strftime('%Y%m%d_%H%M')
            #parquet_path = os.path.join(path, f'{current_date}_{tsfresh_filename}')
            #features_df.to_parquet(parquet_path)
            ##print(f"特征数据已保存为 {parquet_path}")
        except Exception as e:
            print("Error during saving to Parquet:", e)
            return pl.DataFrame()

        # 将 pandas DataFrame 转换为 polars DataFrame 后返回
        try:
            #return pl.from_pandas(features_df)
            #print("传回数据啦")
            return features_df,self.all_columns
        except Exception as e:
            print("Error during converting pandas DataFrame to polars DataFrame:", e)
            return pl.DataFrame()

    def compute_features_with_tsfresh_predict(self, days, outdays, tsfresh_filename,pre_time):

        sn_list=[]
        time_list=[]
        #time1=time.time()
        # 确保 'createtime' 是 datetime 类型
        self.df = self.df.with_columns(
            pl.col('createtime').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
        )
        # 重置索引 (Polars 不支持直接重置索引，但可以通过添加行号)
        df_reset = self.df.with_row_count("index")

        # 确保 'sn' 被视为字符串类型
        if 'sn' in df_reset.columns:
            df_reset = df_reset.with_columns(pl.col('sn').cast(pl.Utf8))
        else:
            #print("Error: 'sn' 列不存在于 DataFrame 中。")
            return pl.DataFrame()

        # 选择数值列（排除 'sn' 列）
        numeric_cols = [
        col for col, dtype in zip(df_reset.columns, df_reset.dtypes)
        if dtype in pl.NUMERIC_DTYPES and col not in ('sn', 'createtime')]
        ##print("numeric_cols",numeric_cols)
        # 初始化用于收集结果的列表
        feature_data = []

        # 定义自定义特征提取参数
        custom_fc_parameters = {
            "mean": None, "median": None, "minimum": None, "maximum": None,
            "standard_deviation": None, "variance": None, "skewness": None, "kurtosis": None,
            "first_location_of_maximum": None, "first_location_of_minimum": None,
            "absolute_sum_of_changes": None, "count_above_mean": None, "count_below_mean": None,
            "autocorrelation": [{"lag": lag} for lag in range(1, 4)],
            "partial_autocorrelation": [{"lag": lag} for lag in range(1, 4)],
            "fft_coefficient": [
                {"coeff": 1, "attr": "real"}, {"coeff": 1, "attr": "imag"},
                {"coeff": 2, "attr": "real"}, {"coeff": 2, "attr": "imag"}
            ],
            "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in range(1, 3)],
            "number_peaks": [{"n": 1}, {"n": 3}],
            "number_cwt_peaks": [{"n": 1}, {"n": 3}],
            "binned_entropy": [{"max_bins": 10}],
            "linear_trend": [{"attr": attr} for attr in ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']],
            "quantile": [{"q": q} for q in [0.1, 0.25, 0.5, 0.75, 0.9]],
            "range_count": [{"min": -1, "max": 1}],
        }

        import time
        start_time1 = time.time()

        # 使用 partition_by 来迭代每个 'sn' 组
        groups = df_reset.partition_by("sn")

        for group_df in groups:
            # 获取当前组的 'sn' 值
            
            sn = group_df["sn"][0]

            # 获取组内的最大 createtime
            new_time_series = group_df.select(pl.col('createtime').max()).to_dict(as_series=False)
            if 'createtime' not in new_time_series or not new_time_series['createtime']:
                continue  # 如果没有 createtime，跳过

            new_time = new_time_series['createtime'][0]
            if isinstance(new_time, datetime):
                # already a datetime object
                pass
            else:
                # 如果 new_time 不是 datetime 对象，根据实际情况进行转换
                # 这里假设 new_time 是一个字符串，您可以根据需要调整
                new_time = datetime.strptime(new_time, "%Y-%m-%d %H:%M:%S")

            # 计算时间窗口
            if new_time>pre_time:
                new_time=pre_time
            end_time = new_time - timedelta(days=days)
            start_time_window = end_time - timedelta(days=outdays)

            sn_list.append(sn)
            time_list.append(end_time)
            #sn #end_time
            #time_json={
            #    'sn':sn,
            #    'end_time':end_time
            #}

            # 在时间窗口内过滤数据
            window_data = group_df.filter(
                (pl.col('createtime') >= start_time_window) & (pl.col('createtime') < end_time)
            )
            if isinstance(window_data, pl.DataFrame):
                window_data1 = window_data.lazy()
            unique_dates_count = window_data1.with_columns(
                pl.col('createtime').dt.truncate('1d').alias('date')
            ).select(pl.col('date').n_unique())

            # 使用 collect() 执行 LazyFrame 计算并获取实际结果
            unique_dates_count_value = unique_dates_count.collect().to_numpy()[0][0]
            #print(f"开始{start_time_window}")
            #print(f"结束{end_time}")
            #print(f"{sn}的天数{unique_dates_count_value}")
            #print(len(window_data))
            if unique_dates_count_value<15:
                
                continue
            if not window_data.is_empty():
                feature_data.append(window_data)
        end_time1 = time.time()
        ##print(f"消耗时间：{end_time1 - start_time1} 秒")

        # 检查是否有可用的特征数据
        if len(feature_data) == 0:
            #print("没有可用的数据在指定的时间窗口内。")
            return pl.DataFrame()  # 如果没有数据，返回空的 DataFrame

        # 合并所有特征数据
        try:
            all_data = pl.concat(feature_data,how="diagonal")
        except Exception as e:
            #print("Error during concatenating feature data:", e)
            return pl.DataFrame()

        ##print("合并后的数据预览：")
        ##print(all_data.head())

        # 确保 'sn' 和 'createtime' 存在于 all_data 中
        #if not {'sn', 'createtime'}.issubset(all_data.columns):
        #    #print("Error: 'sn' 或 'createtime' 列在合并后的数据中缺失。")
        #    return pl.DataFrame()

        # 使用 tsfresh 需要的数据格式，将数据从宽格式转换为长格式
        try:
            melted = all_data.melt(
                id_vars=['sn', 'createtime'],
                value_vars=numeric_cols,
                variable_name='kind',
                value_name='value'
            )
        except Exception as e:
            #print("Error during melting the dataframe:", e)
            return pl.DataFrame()
        time2=time.time()
        ##print(f"花费了{time2-time1}时间！")
        ##print("三小")
        ##print("转换后的长格式数据预览：")
        ##print(melted.head())

        # 使用自定义参数提取特征
        try:
            ##print("三小")
            extracted_features = extract_features(
                melted.to_pandas(),  # tsfresh 目前不支持 polars，因此需要转换为 pandas
                column_id='sn',
                column_sort='createtime',
                column_kind='kind',
                column_value='value',
                default_fc_parameters=custom_fc_parameters,
                disable_progressbar=True
                #n_jobs=-1
            )
            ##print("结束")
        except Exception as e:
            #print("Error during feature extraction:", e)
            return pl.DataFrame()

        ##print("提取的特征预览：")
        ##print(extracted_features.head())

        # 填补缺失值
        try:
            ##print("传回数据啦")
            features_df = impute(extracted_features).reset_index()
        except Exception as e:
            #print("Error during imputing missing values:", e)
            return pl.DataFrame()

        # 重命名列
        features_df = features_df.rename(columns={'index': 'sn'})
        features_df = features_df.loc[:, ~features_df.columns.str.contains('index')]

        ##print("填补缺失值后的特征数据预览：")
        ##print(features_df.head())

        # 将特征保存为 Parquet 文件
        try:
            path = "./data/tsfresh_feature"
            os.makedirs(path, exist_ok=True)
            current_date = datetime.now().strftime('%Y%m%d_%H%M')
            parquet_path = os.path.join(path, f'{current_date}_{tsfresh_filename}')
            #features_df.to_parquet(parquet_path)
            ##print(f"特征数据已保存为 {parquet_path}")
        except Exception as e:
            #print("Error during saving to Parquet:", e)
            return pl.DataFrame()

        # 将 pandas DataFrame 转换为 polars DataFrame 后返回
        try:
            #return pl.from_pandas(features_df)
            #print("传回数据啦")
            return features_df,self.all_columns,sn_list,time_list
        except Exception as e:
            #print("Error during converting pandas DataFrame to polars DataFrame:", e)
            return pl.DataFrame()
        
    def compute_features_with(self, days, outdays, tsfresh_filename):
        sn_list=[]
        time_list=[]
        #time1=time.time()
        # 确保 'createtime' 是 datetime 类型
        try:
            self.df = self.df.with_columns(
                pl.col('createtime').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            )
        except:
            self.df = self.df.with_columns(pl.col("createtime").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S.%f").dt.strftime("%Y-%m-%d %H:%M:%S"))  # 格式化为第一种格式)


        #self.df = self.df.with_columns(
        #pl.col('createtime').str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S.%f")
    #)
        # 重置索引 (Polars 不支持直接重置索引，但可以通过添加行号)
        df_reset = self.df.with_row_count("index")

        # 确保 'sn' 被视为字符串类型
        if 'sn' in df_reset.columns:
            df_reset = df_reset.with_columns(pl.col('sn').cast(pl.Utf8))
        else:
            #print("Error: 'sn' 列不存在于 DataFrame 中。")
            return pl.DataFrame()
        df_reset = df_reset.drop("index")
        #print(")00000000")
        # 选择数值列（排除 'sn' 列）
        numeric_cols = [
        col for col, dtype in zip(df_reset.columns, df_reset.dtypes)
        if dtype in pl.NUMERIC_DTYPES and col not in ('sn', 'createtime')]
        agg_expr = []
        df_reset = df_reset.with_columns([pl.col(col).cast(pl.Float64) for col in numeric_cols])
        for col in numeric_cols:
            if col != 'label':
                agg_expr.append(pl.col(col).mean().alias(f'{col}_mean'))
                agg_expr.append(pl.col(col).median().alias(f'{col}_median'))
                agg_expr.append(pl.col(col).min().alias(f'{col}_min'))
                agg_expr.append(pl.col(col).max().alias(f'{col}_max'))
                agg_expr.append(pl.col(col).var().alias(f'{col}_variance'))
                agg_expr.append(pl.col(col).std().alias(f'{col}_std'))
                agg_expr.append(pl.col(col).skew().alias(f'{col}_skew'))
                agg_expr.append(pl.col(col).kurtosis().alias(f'{col}_kurtosis'))
                agg_expr.append(pl.col(col).ts.first_location_of_maximum().alias(f'{col}_flomax'))
                agg_expr.append(pl.col(col).ts.first_location_of_minimum().alias(f'{col}_flomin'))
                agg_expr.append(pl.col(col).ts.absolute_sum_of_changes().alias(f'{col}_asoc'))
                agg_expr.append(pl.col(col).ts.count_above_mean().alias(f'{col}_cam'))
                agg_expr.append(pl.col(col).ts.count_below_mean().alias(f'{col}_cbm'))
                #for i in range(1,4):
                #    agg_expr.append(pl.col(col).ts.autocorrelation(n_lags=i).alias(f'{col}_ac_lag{i}'))
                #    # TODO partial auto
                #    agg_expr.append(pl.col(col).map_elements(partial_auto(i).add).alias(f'{col}_pa{i}'))
                # FFT
                agg_expr.append(pl.col(col).map_elements(fft).alias(f'{col}_fft'))
                for i in range(1,3):
                    agg_expr.append(pl.col(col).ts.time_reversal_asymmetry_statistic(n_lags=i).alias(f'{col}_tras_lag{i}'))
                agg_expr.append(pl.col(col).ts.number_peaks(support=1).alias(f'{col}_np_1'))
                agg_expr.append(pl.col(col).ts.number_peaks(support=3).alias(f'{col}_np_3'))
                agg_expr.append(pl.col(col).ts.binned_entropy(bin_count=10).alias(f'{col}_be'))
                agg_expr.append(pl.col(col).ts.linear_trend().name.prefix(f"{col}"))#attention
                for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    agg_expr.append(pl.col(col).quantile(quantile=q).alias(f'{col}_quantile_{q}'))
                agg_expr.append(pl.col(col).ts.range_count(lower=-1, upper=1).alias(f'{col}_range_count'))

            #if col[0]=="r":
                #agg_expr.append(pl.col(col).std().alias(f'{col}_std'))
        
        # 使用 partition_by 来迭代每个 'sn' 组
        
        feature_data = []
        groups = df_reset.partition_by("sn")
        #print("=============")
        df_a=pl.DataFrame()

        for group_df in groups:

            # 获取当前组的 'sn' 值
            sn = group_df["sn"][0]
            # 获取组内的最大 createtime
            new_time_series = group_df.select(pl.col('createtime').max()).to_dict(as_series=False)
            
            if 'createtime' not in new_time_series or not new_time_series['createtime']:
                continue  # 如果没有 createtime，跳过
            new_time = new_time_series['createtime'][0]
            if isinstance(new_time, datetime):
                # already a datetime object
                pass
            else:
                # 如果 new_time 不是 datetime 对象，根据实际情况进行转换
                # 这里假设 new_time 是一个字符串，您可以根据需要调整
                new_time = datetime.strptime(new_time, "%Y-%m-%d %H:%M:%S")
            # 计算时间窗口
                group_df = group_df.with_columns(
                    pl.col("createtime").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S.%f"))

# 如果需要再格式化为字符串类型
                group_df = group_df.with_columns(pl.col("createtime").dt.strftime("%Y-%m-%d %H:%M:%S"))
            end_time = new_time - timedelta(days=days)
            start_time_window = end_time - timedelta(days=outdays)
            print('group_df',group_df["createtime"])
            #try:
            # 在时间窗口内过滤数据
            window_data = group_df.filter(
                    (pl.col('createtime') >= start_time_window) & (pl.col('createtime') < end_time)
                )
                #print(len(window_data))
            if not window_data.is_empty():
                feature_data.append(window_data)

        #print("0000",feature_data)
        end_time1 = time.time()
        ##print(f"消耗时间：{end_time1 - start_time1} 秒")

        # 检查是否有可用的特征数据
        print(len(feature_data))
        if len(feature_data) == 0:
            #print("没有可用的数据在指定的时间窗口内。")
            return pl.DataFrame()  # 如果没有数据，返回空的 DataFrame

        # 合并所有特征数据
        #
        #try:
        #print("0.0.0.0.000")
        df_a = pl.concat(feature_data,how="diagonal")
        #print("99999999")
        #print(df_a)
        #except Exception as e:
            #print("Error during concatenating feature data:", e)
            #return pl.DataFrame()
        
        try:
            #df_new = df_a.with_columns(pl.col('createtime').cast(pl.Datetime).dt.date().alias('date'))
            #result = df_new.group_by(['sn', 'date']).agg(pl.col('createtime').n_unique()).sort(by=['sn', 'date'],descending=True)

            #sn_date_last_day = result.filter(pl.col('createtime') > 100).group_by('sn').agg(pl.col('date').max())
            #sn_date_last_day = result.filter(pl.col('createtime') > 100).group_by('sn').agg(pl.all())
            #not_melted_lastday = df_new.join(sn_date_last_day, on=['sn','date'], how='inner')
            #return pl.from_pandas(features_df)
            #print("传回数据啦")
            #print('0.0.0.0',len(not_melted_lastday))
            print('长度',len(df_a))
            not_melted_lastday=df_a.sort(by=['sn', 'createtime'], descending=True).group_by('sn').agg(agg_expr)
            not_melted_lastday=split_structs(not_melted_lastday,numeric_cols)
            
            #print('99999999999999999999',not_melted_lastday)
            #print(self.all_columns)
            return not_melted_lastday ,self.all_columns
        except Exception as e:
            print("Error during converting pandas DataFrame to polars DataFrame:", e)

            import traceback
            traceback.print_exc()
            return pl.DataFrame()
        
    
    def compute_features_with_predict(self, days, outdays, tsfresh_filename,pre_time):
        sn_list=[]
        time_list=[]
        #time1=time.time()
        # 确保 'createtime' 是 datetime 类型
        self.df = self.df.with_columns(
            pl.col('createtime').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
        )
        # 重置索引 (Polars 不支持直接重置索引，但可以通过添加行号)
        df_reset = self.df.with_row_count("index")

        # 确保 'sn' 被视为字符串类型
        if 'sn' in df_reset.columns:
            df_reset = df_reset.with_columns(pl.col('sn').cast(pl.Utf8))
        else:
            #print("Error: 'sn' 列不存在于 DataFrame 中。")
            return pl.DataFrame()
        df_reset = df_reset.drop("index")

        print("88888888888")
        # 选择数值列（排除 'sn' 列）
        numeric_cols = [
        col for col, dtype in zip(df_reset.columns, df_reset.dtypes)
        if dtype in pl.NUMERIC_DTYPES and col not in ('sn', 'createtime')]
        agg_expr = []
        df_reset = df_reset.with_columns([pl.col(col).cast(pl.Float64) for col in numeric_cols])
        for col in numeric_cols:
            if col != 'label' :
                agg_expr.append(pl.col(col).mean().alias(f'{col}_mean'))
                agg_expr.append(pl.col(col).median().alias(f'{col}_median'))
                agg_expr.append(pl.col(col).min().alias(f'{col}_min'))
                agg_expr.append(pl.col(col).max().alias(f'{col}_max'))
                agg_expr.append(pl.col(col).var().alias(f'{col}_variance'))
                agg_expr.append(pl.col(col).std().alias(f'{col}_std'))
                agg_expr.append(pl.col(col).skew().alias(f'{col}_skew'))
                agg_expr.append(pl.col(col).kurtosis().alias(f'{col}_kurtosis'))
                agg_expr.append(pl.col(col).ts.first_location_of_maximum().alias(f'{col}_flomax'))
                agg_expr.append(pl.col(col).ts.first_location_of_minimum().alias(f'{col}_flomin'))
                agg_expr.append(pl.col(col).ts.absolute_sum_of_changes().alias(f'{col}_asoc'))
                agg_expr.append(pl.col(col).ts.count_above_mean().alias(f'{col}_cam'))
                agg_expr.append(pl.col(col).ts.count_below_mean().alias(f'{col}_cbm'))
                #for i in range(1,4):
                #    agg_expr.append(pl.col(col).ts.autocorrelation(n_lags=i).alias(f'{col}_ac_lag{i}'))
                #    # TODO partial auto
                #    agg_expr.append(pl.col(col).map_elements(partial_auto(i).add).alias(f'{col}_pa{i}'))
                # FFT
                agg_expr.append(pl.col(col).map_elements(fft).alias(f'{col}_fft'))
                for i in range(1,3):
                    agg_expr.append(pl.col(col).ts.time_reversal_asymmetry_statistic(n_lags=i).alias(f'{col}_tras_lag{i}'))
                agg_expr.append(pl.col(col).ts.number_peaks(support=1).alias(f'{col}_np_1'))
                agg_expr.append(pl.col(col).ts.number_peaks(support=3).alias(f'{col}_np_3'))
                agg_expr.append(pl.col(col).ts.binned_entropy(bin_count=10).alias(f'{col}_be'))
                agg_expr.append(pl.col(col).ts.linear_trend().name.prefix(f"{col}"))#attention
                for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    agg_expr.append(pl.col(col).quantile(quantile=q).alias(f'{col}_quantile_{q}'))
                agg_expr.append(pl.col(col).ts.range_count(lower=-1, upper=1).alias(f'{col}_range_count'))

                #if col[0]=="r":
                #    agg_expr.append(pl.col(col).std().alias(f'{col}_std'))

        # 使用 partition_by 来迭代每个 'sn' 组
        feature_data = []
        groups = df_reset.partition_by("sn")

        for group_df in groups:
            # 获取当前组的 'sn' 值
            sn = group_df["sn"][0]
            # 获取组内的最大 createtime
            new_time_series = group_df.select(pl.col('createtime').max()).to_dict(as_series=False)
            if 'createtime' not in new_time_series or not new_time_series['createtime']:
                continue  # 如果没有 createtime，跳过
            new_time = new_time_series['createtime'][0]
            if isinstance(new_time, datetime):
                # already a datetime object
                pass
            else:
                # 如果 new_time 不是 datetime 对象，根据实际情况进行转换
                # 这里假设 new_time 是一个字符串，您可以根据需要调整
                new_time = datetime.strptime(new_time, "%Y-%m-%d %H:%M:%S")
            # 计算时间窗口
            if new_time>pre_time:
                new_time=pre_time
            end_time = new_time - timedelta(days=days)
            start_time_window = end_time - timedelta(days=outdays)
            sn_list.append(sn)
            time_list.append(end_time)
            # 在时间窗口内过滤数据
            window_data = group_df.filter(
                (pl.col('createtime') >= start_time_window) & (pl.col('createtime') < end_time)
            )

            if not window_data.is_empty():
                feature_data.append(window_data)
        end_time1 = time.time()
        ##print(f"消耗时间：{end_time1 - start_time1} 秒")

        # 检查是否有可用的特征数据
        if len(feature_data) == 0:
            #print("没有可用的数据在指定的时间窗口内。")
            return pl.DataFrame()  # 如果没有数据，返回空的 DataFrame

        # 合并所有特征数据
        try:
            df_a = pl.concat(feature_data,how="diagonal")
        except Exception as e:
            print("Error during concatenating feature data:", e)
            return pl.DataFrame()
        
        try:
            print('长度',len(df_a))
            not_melted_lastday=df_a.sort(by=['sn', 'createtime'], descending=True).group_by('sn').agg(agg_expr)
            print('09090jc',len(not_melted_lastday))
            not_melted_lastday=split_structs(not_melted_lastday,numeric_cols)
            
            #print('99999999999999999999',not_melted_lastday)
            #print(self.all_columns)
            return not_melted_lastday ,self.all_columns,sn_list,time_list,self.sn_na_list
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Error during converting pandas DataFrame to polars DataFrame:", e)
            return pl.DataFrame()
        
        #result_ng = not_melted_lastday_ng.group_by('sn').agg(agg_expr)

