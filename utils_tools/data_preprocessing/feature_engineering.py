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
from datetime import date
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

class featureProcessor:
    @log_execution_time
    def __init__(self, df: pl.DataFrame,all_columns):
        #self.df = df
        #添加数据处理模块
        self.df,self.all_columns,self.sn_na_list=DataProcessor(df,all_columns).main()
    

    def compute_features_with(self, out_days,time_range,tsfresh_filename):
        visible_date=None
        vpv_noise_threshold = 200
        ipv_threshold = 5000
        vpv_default = 1050
        vpv_spike_amplitude = [0.8, 0.9, 1.0, 1.1, 1.2]
        vpv_sag_amplitude = [0.05, 0.1]
        vac_default = 220
        vac_spike_amplitude = [1.1, 1.2, 1.3]
        vac_sag_amplitude = [0.8, 0.9, 1.0]        

        df = self.df.with_columns(
        pl.col('createtime').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
        )
        #df = df.with_columns(pl.col('errorcode').cast(pl.Float64))
        df = df.with_columns(
        pl.col('errorcode').cast(pl.Utf8)  # 将列转换为字符串类型
        )


        sn_list=[]
        time_list=[]

        groups = df.partition_by("sn")
        feature_list = []
        for group_df in groups:
            sn = group_df['sn'][0]
            sn_list.append(sn)
            group_df = group_df.sort('createtime')
            # 根据date判断可利用的最后一天数据,然后剔除out_day天的数据
            # 最后仅保留time_range天的数据用于特征生成
            if visible_date:
                    #print("===========")
                    #print(visible_date)
                try:
                    visible_date = visible_date.date()
                except:
                    visible_date
                last_date = min(group_df['createtime'].dt.date().max(), visible_date) - timedelta(days=out_days-1)
            else:
                last_date = group_df['createtime'].dt.date().max() - timedelta(days=out_days-1)
            start_date = last_date - timedelta(days=time_range)
            group_df_filtered = group_df.filter((pl.col('createtime') > start_date) &
                                                (pl.col('createtime') < last_date))
            # 构建日期列
            time_list.append(last_date)
            group_df_filtered = group_df_filtered.with_columns(
                pl.col('createtime').dt.date().alias("date")
            )

            # 构建error_code有无状态
            group_df_filtered = group_df_filtered.with_columns(
                (~pl.col("errorcode").is_in(["0"])).alias("error_status")
            )
            # 构建日度error_rate,on_rate,eday特征
            agg_expr = []
            agg_expr.append(pl.col("error_status").mean().alias('error_rate'))
            agg_expr.append((pl.col("errorcode")!="0").any().alias('error_occur'))
            agg_expr.append(pl.col("switchstatus").mean().alias('on_rate'))
            agg_expr.append(pl.col('sn').first().alias('sn'))
            for i in range(1,7):
                agg_expr.append((pl.col(f"ipv{i}")>ipv_threshold).any().alias(f"ipv{i}_peaks"))
            for i in range(1,7):
                print(vpv_spike_amplitude,vpv_default)
                for amp in vpv_spike_amplitude:
                    agg_expr.append((pl.col(f"vpv{i}")>vpv_default*amp).sum().alias(f"vpv{i}_{amp}_spikes"))
                for amp in vpv_sag_amplitude:
                    agg_expr.append((pl.col(f"vpv{i}")<vpv_default*amp).sum().alias(f"vpv{i}_{amp}_sags"))
            for i in range(1,4):
                for amp in vac_spike_amplitude:
                    agg_expr.append((pl.col(f"vac{i}")>vac_default*amp).sum().alias(f"vac{i}_{amp}_spikes"))
                for amp in vac_sag_amplitude:
                    agg_expr.append((pl.col(f"vac{i}")<vac_default*amp).sum().alias(f"vac{i}_{amp}_sags"))
            iac_threshold = int(sn[2:4])*1000/220/1.732/3
            for i in range(1,4):agg_expr.append((pl.col(f"iac{i}")>iac_threshold).sum().alias(f"iac{i}_peaks"))
            print(group_df_filtered)
            print(group_df_filtered['date'])
            result_1 = group_df_filtered.group_by('date').agg(agg_expr)
            # 填充上所有日期的数据,若数据缺失用后一天的数据填充
            date_df = pl.DataFrame({'date':pl.date_range(start=start_date, end=last_date, interval="1d", closed="left", eager=True)})
            result_1 = date_df.join(result_1, on = 'date', how = 'left').sort('date')

            # 在剔除switch_status为0（交流断路）的情况下,生成日度vpv, vac特征
            target_cols = ([f"vac{i}" for i in range(1,4)]
                         + [f"vpv{i}" for i in range(1,7)]
                          )
            agg_expr = []
            for col in target_cols:
                agg_expr.append(pl.col(col).mean().alias(f'{col}_mean'))
                agg_expr.append(pl.col(col).var().alias(f'{col}_variance'))
            result_2 = group_df_filtered.filter(pl.col('switchstatus')==1).group_by('date').agg(agg_expr)

            # 在仅仅考虑11:00-15:00数据的情况下(确保取出来为满负荷工作状态的temperature)
            # 同时,也剔除switchstatus为0的数据
            # 生成temperature特征
            result_3 = (
                group_df_filtered
                .filter(pl.col('switchstatus')==1)
                .filter((pl.col('createtime').dt.hour() >= 11) & (pl.col('createtime').dt.hour() < 15))
                .group_by('date')
                .agg(pl.col('temperature').mean().alias('temperature_mean'))
                )

            # 从业务侧得知,早晚eday计算存在误差,仅考虑9:00-15:00的eday更准确
            result_4 = (
                group_df_filtered
                .filter((pl.col('createtime').dt.hour() >= 9) & (pl.col('createtime').dt.hour() < 15))
                .group_by('date')
                .agg([pl.col('eday').first().alias('eday_start'),
                      pl.col('eday').last().alias('eday_end'),
                      pl.col("switchstatus").mean().alias('on_rate_4_eday')])
                )

            # result_1, result_2, result_3结果合并
            # !!!TODO:注意fill_null的效果是否会影响结果
            result = result_1.join(result_2, on = 'date', how = 'left')
            result = result.join(result_3, on = 'date', how = 'left')
            result = result.join(result_4, on = 'date', how = 'left').sort('date').fill_null(strategy="backward").fill_null(0)

            def percentage_difference(data):
                """若分母为0,则替换成1,以消除inf
                """
                numerator = (data.diff()[1:]).to_numpy()
                denominator = (data[:-1]).to_numpy()

                output = np.divide(numerator, denominator, where=(denominator!=0))
                return output.tolist()

            # 产生vpv特征
            vpv_mean = {}
            vpv_variance = {}
            for i in range(1,7):
                if result[f"vpv{i}_mean"].max() < vpv_noise_threshold:
                    vpv_mean[i] = [0]*(time_range-1)
                    vpv_variance[i] = [0]*(time_range-1)
                else:
                    vpv_mean[i] = percentage_difference(result[f"vpv{i}_mean"])
                    vpv_variance[i] = percentage_difference(result[f"vpv{i}_variance"])
            # 产生vac特征
            vac_mean = {}
            vac_variance = {}
            for i in range(1,4):
                vac_mean[i] = percentage_difference(result[f"vac{i}_mean"])
                vac_variance[i] = percentage_difference(result[f"vac{i}_variance"])
            # 产生温度特征
            temperature = result["temperature_mean"].to_list()
            temperature_diff = percentage_difference(result["temperature_mean"])
            # 产生eday特征
            eday = np.divide((result['eday_end'] - result['eday_start']), result['on_rate_4_eday'],
                             where=(result['on_rate_4_eday'].to_numpy()!=0)) 
            eday = percentage_difference(eday)
            # error_code比例特征
            error_occur = result["error_occur"].to_list()
            error_rate = result["error_rate"].to_list()
            # switch_status比例特征
            on_rate = result["on_rate"].to_list()
            # ipv peaks特征
            ipv_peaks = {}
            iac_peaks = {}
            vpv_spikes = {}
            vac_spikes = {}
            vpv_sags = {}
            vac_sags = {}
            for i in range(1,7):
                ipv_peaks[i] = result[f"ipv{i}_peaks"].to_list()
                for amp in vpv_spike_amplitude:
                    vpv_spikes[f"{i}_{amp}"] = result[f"vpv{i}_{amp}_spikes"].to_list()
                for amp in vpv_sag_amplitude:
                    vpv_sags[f"{i}_{amp}"] = result[f"vpv{i}_{amp}_sags"].to_list()
            for i in range(1,4):
                iac_peaks[i] = result[f"iac{i}_peaks"].to_list()
                for amp in vac_spike_amplitude:
                    vac_spikes[f"{i}_{amp}"] = result[f"vac{i}_{amp}_spikes"].to_list()
                for amp in vac_sag_amplitude:
                    vac_sags[f"{i}_{amp}"] = result[f"vac{i}_{amp}_sags"].to_list()

            features = [sn]
            for i in range(1,7):features += vpv_mean[i]
            for i in range(1,7):features += vpv_variance[i]
            for i in range(1,4):features += vac_mean[i]
            for i in range(1,4):features += vac_variance[i]
            for i in range(1,7):features += ipv_peaks[i]
            for i in range(1,4):features += iac_peaks[i]
            for i in range(1,7):
                for amp in vpv_spike_amplitude:
                    features += vpv_spikes[f"{i}_{amp}"]
                for amp in vpv_sag_amplitude:
                    features += vpv_sags[f"{i}_{amp}"]
            for i in range(1,4):
                for amp in vac_spike_amplitude:
                    features += vac_spikes[f"{i}_{amp}"]
                for amp in vac_sag_amplitude:
                    features += vac_sags[f"{i}_{amp}"]
            features += temperature + temperature_diff
            features += eday + error_occur + error_rate + on_rate
            feature_list.append(features)
    
        # 生成特征列名,并且拼接所有特征
        feature_names = (["sn"] + 
                         [f"vpv{i}_mean_day_{j}" for i in range(1,7) for j in range(1, time_range)] +
                         [f"vpv{i}_variance_day_{j}" for i in range(1,7) for j in range(1, time_range)] +
                         [f"vac{i}_mean_day_{j}" for i in range(1,4) for j in range(1, time_range)] +
                         [f"vac{i}_variance_day_{j}" for i in range(1,4) for j in range(1, time_range)] +
                         [f"ipv{i}_peaks_day_{j}" for i in range(1,7) for j in range(time_range)] +
                         [f"iac{i}_peaks_day_{j}" for i in range(1,4) for j in range(time_range)] +

                         [f"vpv{i}_{amp}_spikes_day_{j}" for i in range(1,7) for amp in vpv_spike_amplitude for j in range(time_range)] +
                         [f"vpv{i}_{amp}_sags_day_{j}" for i in range(1,7) for amp in vpv_sag_amplitude for j in range(time_range)] +
                         [f"vac{i}_{amp}_spikes_day_{j}" for i in range(1,4) for amp in vac_spike_amplitude for j in range(time_range)] +
                         [f"vac{i}_{amp}_sags_day_{j}" for i in range(1,4) for amp in vac_sag_amplitude for j in range(time_range)] +

                         [f"temperature_day_{j}" for j in range(time_range)] +
                         [f"temperature_diff_day_{j}" for j in range(1, time_range)] +
                         [f"eday_day_{j}" for j in range(1, time_range)] +
                         [f"error_occur_day_{j}" for j in range(time_range)] +
                         [f"error_rate_day_{j}" for j in range(time_range)] +
                         [f"on_rate_day_{j}" for j in range(time_range)]
                        )
        feature_df = pl.DataFrame(feature_list,
                                  schema=feature_names)

      
        
        path = "./data/tsfresh_feature"
        os.makedirs(path, exist_ok=True)
        current_date = datetime.now().strftime('%Y%m%d_%H%M')
        parquet_path = os.path.join(path, f'{current_date}_{tsfresh_filename}')
        #parquet_path='ok.csv'
        feature_df.write_parquet(parquet_path)
        #print(f"特征数据已保存为 {parquet_path}")
        #print('2222',feature_df)
        return feature_df,self.all_columns

    def compute_features_with_predict(self,out_days, time_range,tsfresh_filename,visible_date):
        """特征工程
    基于业务信息，产生如下特征：
        1. 归一化的vpv的日间差异
        2. 归一化的vac的日间差异
        3. 温度的日间差异(temperature_occur)和每日温度(temperature)
        4. eday的日间差异
        5. errorcode非0的比例(error_rate)和是否当天有errorcode(error_occur)
        6. switch_status为1的比例(on_Rate)
        7. ipv的超高数据是否存在(ipv_peaks)
        8. vpv的超高数据发生次数(vpv_peaks)
        9. iac的超高数据发生次数(iac_peaks)

    计算逻辑：
        1. 将cratetime列转换为时间特征,并且将df按sn拆分
        2. 按visible_date(若有)或者sn存在的最后一个日期确定截止日期,再将截至日期往前推out_day天,
        取time_range天的数据到此日期
        2. 在所有数据上,计算每天的error_rate,error_occur,on_rate,ipv_peaks,vpv_peaks,iac_peaks
        3. 在剔除开路情况下(仅考虑switchstatus==1),计算每日的vpv, vac的平均和方差
        4. 计算11:00-15:00的平均温度,确保取到的是满负荷工作温度
        5. 计算09:00-15:00的发电量,业务了解得知早晚发电计算存在误差
        6. 拼接结果,若存在某天维度NA,用后一天的数据拼接,若无后一天数据,用0赋值
        7. 对取差分的量，进行差分比例计算,若出现分母为0,则将结果替换为0
        8. 聚合结果特征
    
    输入:
        df: 数据矩阵,要求必须要有ipv1,...,ipv6,vpv1,...,vpv6,vac1,vac2,vac3,temperature,errorcode,
        switchstatus,sn,createtime
        visiblie_date: 人为设定自visible_date后的数据不可见
        out_days: 人为设定忽略out_days天的数据
        time_range: 计算特征时考虑的天数
        vpv_noise_threshold: vpv未接通时存在感电, 一般感电电压低于200V
        ipv_theshold: ipv数据偶发异常高值,将高于5000A的ipv进行统计
    
    输出:
        feature_df: 特征矩阵,输出维度为num_of_unique_sn * (1 + 2*6*(time_range-1) + 2*3*(time_range-1)
                        + 6*time_range + 6*(len(vpv_spike_amplitude)+len(vpv_sag_amplitude))*time_range
                        + 3*(len(vac_spike_amplitude)+len(vac_sag_amplitude))*time_range + 3*time_range
                        + 4*time_range + 2*(time_range-1))
    """    
        sn_list=[]
        time_list=[]
        vpv_noise_threshold = 200
        ipv_threshold = 5000
        vpv_default = 1050
        vpv_spike_amplitude = [0.8, 0.9, 1.0, 1.1, 1.2]
        vpv_sag_amplitude = [0.05, 0.1]
        vac_default = 220
        vac_spike_amplitude = [1.1, 1.2, 1.3]
        vac_sag_amplitude = [0.8, 0.9, 1.0]        

        df = self.df.with_columns(
        pl.col('createtime').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
        )
        #df = df.with_columns(pl.col('errorcode').cast(pl.Float64))
        df = df.with_columns(
        pl.col('errorcode').cast(pl.Utf8)  # 将列转换为字符串类型
        )


        sn_list=[]
        time_list=[]

        groups = df.partition_by("sn")
        feature_list = []
        for group_df in groups:
            sn = group_df['sn'][0]
            sn_list.append(sn)
            group_df = group_df.sort('createtime')
            # 根据date判断可利用的最后一天数据,然后剔除out_day天的数据
            # 最后仅保留time_range天的数据用于特征生成
            if visible_date:
                    #print("===========")
                    #print(visible_date)
                try:
                    visible_date = visible_date.date()
                except:
                    visible_date
                last_date = min(group_df['createtime'].dt.date().max(), visible_date) - timedelta(days=out_days-1)
            else:
                last_date = group_df['createtime'].dt.date().max() - timedelta(days=out_days-1)
            start_date = last_date - timedelta(days=time_range)
            group_df_filtered = group_df.filter((pl.col('createtime') > start_date) &
                                                (pl.col('createtime') < last_date))
            # 构建日期列
            time_list.append(last_date)
            group_df_filtered = group_df_filtered.with_columns(
                pl.col('createtime').dt.date().alias("date")
            )

            # 构建error_code有无状态
            group_df_filtered = group_df_filtered.with_columns(
                (~pl.col("errorcode").is_in(["0"])).alias("error_status")
            )
            # 构建日度error_rate,on_rate,eday特征
            agg_expr = []
            agg_expr.append(pl.col("error_status").mean().alias('error_rate'))
            agg_expr.append((pl.col("errorcode")!="0").any().alias('error_occur'))
            agg_expr.append(pl.col("switchstatus").mean().alias('on_rate'))
            agg_expr.append(pl.col('sn').first().alias('sn'))
            for i in range(1,7):
                agg_expr.append((pl.col(f"ipv{i}")>ipv_threshold).any().alias(f"ipv{i}_peaks"))
            for i in range(1,7):
                for amp in vpv_spike_amplitude:
                    agg_expr.append((pl.col(f"vpv{i}")>vpv_default*amp).sum().alias(f"vpv{i}_{amp}_spikes"))
                for amp in vpv_sag_amplitude:
                    agg_expr.append((pl.col(f"vpv{i}")<vpv_default*amp).sum().alias(f"vpv{i}_{amp}_sags"))
            for i in range(1,4):
                for amp in vac_spike_amplitude:
                    agg_expr.append((pl.col(f"vac{i}")>vac_default*amp).sum().alias(f"vac{i}_{amp}_spikes"))
                for amp in vac_sag_amplitude:
                    agg_expr.append((pl.col(f"vac{i}")<vac_default*amp).sum().alias(f"vac{i}_{amp}_sags"))
            iac_threshold = int(sn[2:4])*1000/220/1.732/3
            for i in range(1,4):agg_expr.append((pl.col(f"iac{i}")>iac_threshold).sum().alias(f"iac{i}_peaks"))
            result_1 = group_df_filtered.group_by('date').agg(agg_expr)
            # 填充上所有日期的数据,若数据缺失用后一天的数据填充
            date_df = pl.DataFrame({'date':pl.date_range(start=start_date, end=last_date, interval="1d", closed="left", eager=True)})
            result_1 = date_df.join(result_1, on = 'date', how = 'left').sort('date')

            # 在剔除switch_status为0（交流断路）的情况下,生成日度vpv, vac特征
            target_cols = ([f"vac{i}" for i in range(1,4)]
                         + [f"vpv{i}" for i in range(1,7)]
                          )
            agg_expr = []
            for col in target_cols:
                agg_expr.append(pl.col(col).mean().alias(f'{col}_mean'))
                agg_expr.append(pl.col(col).var().alias(f'{col}_variance'))
            result_2 = group_df_filtered.filter(pl.col('switchstatus')==1).group_by('date').agg(agg_expr)

            # 在仅仅考虑11:00-15:00数据的情况下(确保取出来为满负荷工作状态的temperature)
            # 同时,也剔除switchstatus为0的数据
            # 生成temperature特征
            result_3 = (
                group_df_filtered
                .filter(pl.col('switchstatus')==1)
                .filter((pl.col('createtime').dt.hour() >= 11) & (pl.col('createtime').dt.hour() < 15))
                .group_by('date')
                .agg(pl.col('temperature').mean().alias('temperature_mean'))
                )

            # 从业务侧得知,早晚eday计算存在误差,仅考虑9:00-15:00的eday更准确
            result_4 = (
                group_df_filtered
                .filter((pl.col('createtime').dt.hour() >= 9) & (pl.col('createtime').dt.hour() < 15))
                .group_by('date')
                .agg([pl.col('eday').first().alias('eday_start'),
                      pl.col('eday').last().alias('eday_end'),
                      pl.col("switchstatus").mean().alias('on_rate_4_eday')])
                )

            # result_1, result_2, result_3结果合并
            # !!!TODO:注意fill_null的效果是否会影响结果
            result = result_1.join(result_2, on = 'date', how = 'left')
            result = result.join(result_3, on = 'date', how = 'left')
            result = result.join(result_4, on = 'date', how = 'left').sort('date').fill_null(strategy="backward").fill_null(0)

            def percentage_difference(data):
                """若分母为0,则替换成1,以消除inf
                """
                numerator = (data.diff()[1:]).to_numpy()
                denominator = (data[:-1]).to_numpy()

                output = np.divide(numerator, denominator, where=(denominator!=0))
                return output.tolist()

            # 产生vpv特征
            vpv_mean = {}
            vpv_variance = {}
            for i in range(1,7):
                if result[f"vpv{i}_mean"].max() < vpv_noise_threshold:
                    vpv_mean[i] = [0]*(time_range-1)
                    vpv_variance[i] = [0]*(time_range-1)
                else:
                    vpv_mean[i] = percentage_difference(result[f"vpv{i}_mean"])
                    vpv_variance[i] = percentage_difference(result[f"vpv{i}_variance"])
            # 产生vac特征
            vac_mean = {}
            vac_variance = {}
            for i in range(1,4):
                vac_mean[i] = percentage_difference(result[f"vac{i}_mean"])
                vac_variance[i] = percentage_difference(result[f"vac{i}_variance"])
            # 产生温度特征
            temperature = result["temperature_mean"].to_list()
            temperature_diff = percentage_difference(result["temperature_mean"])
            # 产生eday特征
            eday = np.divide((result['eday_end'] - result['eday_start']), result['on_rate_4_eday'],
                             where=(result['on_rate_4_eday'].to_numpy()!=0)) 
            eday = percentage_difference(eday)
            # error_code比例特征
            error_occur = result["error_occur"].to_list()
            error_rate = result["error_rate"].to_list()
            # switch_status比例特征
            on_rate = result["on_rate"].to_list()
            # ipv peaks特征
            ipv_peaks = {}
            iac_peaks = {}
            vpv_spikes = {}
            vac_spikes = {}
            vpv_sags = {}
            vac_sags = {}
            for i in range(1,7):
                ipv_peaks[i] = result[f"ipv{i}_peaks"].to_list()
                for amp in vpv_spike_amplitude:
                    vpv_spikes[f"{i}_{amp}"] = result[f"vpv{i}_{amp}_spikes"].to_list()
                for amp in vpv_sag_amplitude:
                    vpv_sags[f"{i}_{amp}"] = result[f"vpv{i}_{amp}_sags"].to_list()
            for i in range(1,4):
                iac_peaks[i] = result[f"iac{i}_peaks"].to_list()
                for amp in vac_spike_amplitude:
                    vac_spikes[f"{i}_{amp}"] = result[f"vac{i}_{amp}_spikes"].to_list()
                for amp in vac_sag_amplitude:
                    vac_sags[f"{i}_{amp}"] = result[f"vac{i}_{amp}_sags"].to_list()

            features = [sn]
            for i in range(1,7):features += vpv_mean[i]
            for i in range(1,7):features += vpv_variance[i]
            for i in range(1,4):features += vac_mean[i]
            for i in range(1,4):features += vac_variance[i]
            for i in range(1,7):features += ipv_peaks[i]
            for i in range(1,4):features += iac_peaks[i]
            for i in range(1,7):
                for amp in vpv_spike_amplitude:
                    features += vpv_spikes[f"{i}_{amp}"]
                for amp in vpv_sag_amplitude:
                    features += vpv_sags[f"{i}_{amp}"]
            for i in range(1,4):
                for amp in vac_spike_amplitude:
                    features += vac_spikes[f"{i}_{amp}"]
                for amp in vac_sag_amplitude:
                    features += vac_sags[f"{i}_{amp}"]
            features += temperature + temperature_diff
            features += eday + error_occur + error_rate + on_rate
            feature_list.append(features)
    
        # 生成特征列名,并且拼接所有特征
        feature_names = (["sn"] + 
                         [f"vpv{i}_mean_day_{j}" for i in range(1,7) for j in range(1, time_range)] +
                         [f"vpv{i}_variance_day_{j}" for i in range(1,7) for j in range(1, time_range)] +
                         [f"vac{i}_mean_day_{j}" for i in range(1,4) for j in range(1, time_range)] +
                         [f"vac{i}_variance_day_{j}" for i in range(1,4) for j in range(1, time_range)] +
                         [f"ipv{i}_peaks_day_{j}" for i in range(1,7) for j in range(time_range)] +
                         [f"iac{i}_peaks_day_{j}" for i in range(1,4) for j in range(time_range)] +

                         [f"vpv{i}_{amp}_spikes_day_{j}" for i in range(1,7) for amp in vpv_spike_amplitude for j in range(time_range)] +
                         [f"vpv{i}_{amp}_sags_day_{j}" for i in range(1,7) for amp in vpv_sag_amplitude for j in range(time_range)] +
                         [f"vac{i}_{amp}_spikes_day_{j}" for i in range(1,4) for amp in vac_spike_amplitude for j in range(time_range)] +
                         [f"vac{i}_{amp}_sags_day_{j}" for i in range(1,4) for amp in vac_sag_amplitude for j in range(time_range)] +

                         [f"temperature_day_{j}" for j in range(time_range)] +
                         [f"temperature_diff_day_{j}" for j in range(1, time_range)] +
                         [f"eday_day_{j}" for j in range(1, time_range)] +
                         [f"error_occur_day_{j}" for j in range(time_range)] +
                         [f"error_rate_day_{j}" for j in range(time_range)] +
                         [f"on_rate_day_{j}" for j in range(time_range)]
                        )
        feature_df = pl.DataFrame(feature_list,
                                  schema=feature_names)

        path = "./data/tsfresh_feature"
        os.makedirs(path, exist_ok=True)
        current_date = datetime.now().strftime('%Y%m%d_%H%M')
        parquet_path = os.path.join(path, f'{current_date}_{tsfresh_filename}')
        #parquet_path='ok.csv'
        feature_df.write_parquet(parquet_path)
        print(f"特征数据已保存为 {parquet_path}")
        #print('2222',feature_df)
        return feature_df,self.all_columns,sn_list,time_list,self.sn_na_list