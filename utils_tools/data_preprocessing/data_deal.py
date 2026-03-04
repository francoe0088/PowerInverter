'''
Author: franco
Date: 2024-11-26 14:39:17
LastEditors: franco
LastEditTime: 2024-11-27 14:51:24
Description: 
'''
import polars as pl
from functools import reduce
from utils_tools.utilities.wraps_fun import log_execution_time
from sklearn.model_selection import train_test_split

import polars as pl
from functools import reduce
import pickle



class DataProcessor:
    def __init__(self, df: pl.DataFrame,all_columns):
        self.df = df
        self.all_columns=all_columns
    def filter_ipv_vpv_column(self):
        # 如果 all_columns 已经提供，直接使用
        if self.all_columns is not None:
            #print("已有数据了！")
            self.df = self.df.select(self.all_columns)
            self.remaining_ipv_columns = [col for col in self.df.columns if col.startswith('ipv')]
            self.remaining_vpv_columns = [col for col in self.df.columns if col.startswith('vpv')]
            return self.df, self.all_columns
        
        # 获取总行数
        total_len = len(self.df)
        
        # 找到 ipv* 和 vpv* 列
        ipv_columns = [col for col in self.df.columns if col.startswith('ipv')]
        vpv_columns = [col for col in self.df.columns if col.startswith('vpv')]
        all_columns = ipv_columns + vpv_columns
        
        # 逐列检查无效数据并剔除
        # for col in all_columns:
        # # 查找符合条件的无效行
        #     error_df = self.df.filter(
        #         (pl.col(col) <= 0) | (pl.col(col).is_null()) | (pl.col(col) > 6000)
        #     )
            
        #     # 计算错误数据占比
        #     error_percentage = len(error_df) / total_len
            
        #     # 如果错误数据占比超过 98%，删除该列
        #     if error_percentage > 0.99:
        #         self.df = self.df.drop(col)
        #         ##print(f"Column '{col}' has been dropped due to high error percentage ({error_percentage:.2%})")
        self.df = self.df.select(
            ['temperature', 'vpv1', 'vpv2', 'vpv3', 'vpv4', 'vpv5', 'vpv6', 'ipv1', 'ipv2', 'ipv3', 
             'ipv4', 'ipv5', 'ipv6', 'vac1', 'vac2', 'vac3', 'iac1', 'iac2', 'iac3', 'fac1', 'fac2', 
             'pac', 'eday', 'etotal', 'ttotal', 'outputpowerratio', 'activepower', 'reactivepower', 
             'inspectingpower', 'powerfactor', 'createtime', 'sn','errorcode','switchstatus']
        )
        #df = self.df.with_columns(pl.col('errorcode').cast(pl.Int32, strict=False))
        self.remaining_ipv_columns = [col for col in self.df.columns if col.startswith('ipv')]
        self.remaining_vpv_columns = [col for col in self.df.columns if col.startswith('vpv')]
        # 获取剩余的所有列名
        float_columns = [col for col in self.df.columns if col not in ['sn', 'createtime','errorcode']]

    # 转换这些列为 float64
        self.df = self.df.with_columns([
            pl.col(col).cast(pl.Float64) for col in float_columns
        ])
        
        # 保存剩余的列信息到 pickle 文件
        columns_to_save = {
            'all_columns': self.all_columns
        }
        filename = "./output/columns_vpv_ipv/columns_vpv_ipv.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(columns_to_save, f)
        
        #print(f"Columns data saved to {filename}")
        
        return self.df, self.all_columns
    def filter_data_power(self) -> pl.DataFrame:
        """
        过滤数据：去除特定列全部为0的行，去除空值，并去重。
        """
        self.df = self.df.fill_null(0)
        #self.df = self.df.drop_nulls()
        # 去重
        self.df = self.df.unique()
        
        return self.df
    def filter_data_sn(self):
        #print(self.df)
        #只保留sn为TP*的数据
        self.df = self.df.filter(pl.col('sn').str.starts_with('TP'))
        return self.df
    def filter_ipv(self):
        # 将ipv1、ipv2、ipv3中值为6553.5的数据转为0，小于0的值剔除
        #ipv_columns = ['ipv1', 'ipv2', 'ipv3']
        #ipv_columns = [f'ipv{i}' for i in range(1, 3)]
        ipv_columns=self.remaining_ipv_columns
        for col in ipv_columns:
            self.df = self.df.with_columns(
                pl.when(pl.col(col) > 5000).then(0).otherwise(pl.col(col))
                .alias(col)
            )
            #self.df = self.df.filter(pl.col(col) >= 0)  # 剔除小于0的值
        return self.df
    def filter_vpv(self):
        # 将vpv1、vpv2、vpv3中值为6553.5的数据转为0，小于0的值剔除
        #vpv_columns = [f'vpv{i}' for i in range(1, 2)]
        vpv_columns=self.remaining_vpv_columns
        for col in vpv_columns:
            self.df = self.df.with_columns(
                pl.when(pl.col(col)>5000).then(0).otherwise(pl.col(col))
                .alias(col)
            )
        #    self.df = self.df.filter(pl.col(col) >= 0)  # 剔除小于0的值
        return self.df
    def adjust_reactivepower(self):
        # 只有在reactivepower大于16777216时，才减去2的24次方
        self.df = self.df.with_columns(
            pl.when(pl.col('reactivepower') > 16770000)
            .then((pl.col('reactivepower') - 2**24).abs()) 
            .otherwise(pl.col('reactivepower'))  # 否则保持原值
            .alias('reactivepower')
        )
        return self.df
    def adjust_powerfactor(self):
        self.df = self.df.with_columns(
            pl.when(pl.col('powerfactor').pow(2) > 1)  # 使用pow()代替apply()进行平方操作
            .then(pl.col('powerfactor') - 65.536)
            .otherwise(pl.col('powerfactor'))
            .alias('powerfactor')
        )
        return self.df

    def filter_abnormal_sn(self):
        """剔除下列情况的sn:
            1. 3个iac值全为0或null
            2. 6个vpv值全为0或null
            3. 6个ipv值全为0或null
            4. 'outputpowerratio', 'activepower', 'reactivepower', 'inspectingpower'值全为0或null
            5. ttotal值为0或null
            6. eday值为0或null
        """
        agg_expr = []
        for i in range(1,4):
            agg_expr.append(pl.col(f'iac{i}').map_elements(lambda x:set(x.unique())<={0,None}).alias(f'iac{i}_na'))
        for i in range(1,7):
            agg_expr.append(pl.col(f'ipv{i}').map_elements(lambda x:set(x.unique())<={0,None}).alias(f'ipv{i}_na'))
            agg_expr.append(pl.col(f'vpv{i}').map_elements(lambda x:set(x.unique())<={0,None}).alias(f'vpv{i}_na'))
        for col in ['ttotal', 'eday', 'outputpowerratio', 'activepower', 'reactivepower', 'inspectingpower',]:
            agg_expr.append(pl.col(f'{col}').map_elements(lambda x:set(x.unique())<={0,None}).alias(f'{col}_na'))

        result = self.df.group_by('sn').agg(agg_expr)
        sn_iac_na = result.filter((pl.col('iac1_na') == True) &
                                (pl.col('iac2_na') == True) &
                                (pl.col('iac3_na') == True))['sn'].to_list()
        sn_ttotal_na = result.filter((pl.col('ttotal_na') == True))['sn'].to_list()
        sn_eday_na = result.filter((pl.col('eday_na') == True))['sn'].to_list()
        sn_power_na = result.filter((pl.col('outputpowerratio_na') == True) &
                                    (pl.col('activepower_na') == True) &
                                    (pl.col('reactivepower_na') == True) &
                                    (pl.col('inspectingpower_na') == True))['sn'].to_list()
        sn_ipv_na = result.filter((pl.col('ipv1_na') == True) &
                                (pl.col('ipv2_na') == True) &
                                (pl.col('ipv3_na') == True) &
                                (pl.col('ipv4_na') == True) &
                                (pl.col('ipv5_na') == True) &
                                (pl.col('ipv6_na') == True))['sn'].to_list()
        sn_vpv_na = result.filter((pl.col('vpv1_na') == True) &
                                (pl.col('vpv2_na') == True) &
                                (pl.col('vpv3_na') == True) &
                                (pl.col('vpv4_na') == True) &
                                (pl.col('vpv5_na') == True) &
                                (pl.col('vpv6_na') == True))['sn'].to_list()

        # print('ipv有问题的sn为:', sn_ipv_na)
        # print('vpv有问题的sn为:', sn_vpv_na)
        # print('iac有问题的sn为:', sn_iac_na)
        # print('power有问题的sn为:', sn_power_na)
        # print('ttotal有问题的sn为:', sn_ttotal_na)
        # print('eday有问题的sn为:', sn_eday_na)

        sn_na = (set(sn_iac_na) | set(sn_eday_na) | set(sn_ipv_na) |
                set(sn_power_na) | set(sn_ttotal_na) | set(sn_vpv_na))
        
        self.sn_na_list = list(sn_na)
        #print(f"数据异常的sn共{len(sn_na)}个，分别为:", sn_na)
        return self.df.filter(~pl.col('sn').is_in(sn_na))

    def main(self):
        float_columns = [col for col in self.df.columns if col not in ['sn', 'createtime','errorcode']]

    # 转换这些列为 float64
        self.df = self.df.with_columns([
            pl.col(col).cast(pl.Float64) for col in float_columns
        ])
        #columns_to_keep = ['sn', 'createtime']
        #df_zeroed_float = df_zeroed.with_columns([pl.col(col).cast(pl.Float64) for col in columns_to_set_zero])
        self.df = self.filter_data_sn() 
        #self.df = self.filter_abnormal_sn()
        self.df,self.all_columns=self.filter_ipv_vpv_column()
        self.df = self.filter_data_power()  # 步骤 1: 过滤掉特定列全为0的行、去空值并去重
            # 步骤 2: 过滤出sn以TP开头的行
        self.df = self.filter_ipv()         # 步骤 3: ipv1、ipv2、ipv3 的处理
        self.df = self.filter_vpv()         # 步骤 4: vpv1、vpv2、vpv3 的处理
        self.df = self.adjust_reactivepower()  # 步骤 5: 调整reactivepower
        self.df = self.adjust_powerfactor()    # 步骤 6: 调整powerfactor
        self.df = self.filter_abnormal_sn()
        #self.df= self.df.with_columns([pl.all().cast(pl.Float64, strict=False)])
           # 步骤 7: 剔除并报告异常sn
        ##print("6666",self.df)
        return self.df,self.all_columns,self.sn_na_list
    def main1(self):
        
        self.df = self.filter_data_sn() 
        #self.df = self.filter_abnormal_sn()
        self.df,self.all_columns=self.filter_ipv_vpv_column()

        self.df = self.filter_data_power()  # 步骤 1: 过滤掉特定列全为0的行、去空值并去重
            # 步骤 2: 过滤出sn以TP开头的行
        self.df = self.filter_ipv()         # 步骤 3: ipv1、ipv2、ipv3 的处理
        self.df = self.filter_vpv()         # 步骤 4: vpv1、vpv2、vpv3 的处理
        self.df = self.adjust_reactivepower()  # 步骤 5: 调整reactivepower
        self.df = self.adjust_powerfactor()    # 步骤 6: 调整powerfactor
        self.df = self.filter_abnormal_sn()
        #self.df= self.df.with_columns([pl.all().cast(pl.Float64, strict=False)])
           # 步骤 7: 剔除并报告异常sn
        ##print("6666",self.df)
        return self.df,self.all_columns

#数据合并   
import polars as pl
from typing import Optional

def data_combined(data1, data2) -> pl.DataFrame:
    """
    无论 data1, data2 是否为 None，都返回一个列名对齐、类型一致的 DataFrame。
    """
    #if data1 is None and data2 is None:
    #    return pl.DataFrame()
    #elif data1 is None:
    #    # 给 data2 对齐
    #    empty_df = pl.DataFrame({col: pl.Series(name=col, values=[], dtype=pl.Float64) 
    #                             for col in data2.columns})
    #    data1 = empty_df
    #elif data2 is None:
    #    # 给 data1 对齐
    #    empty_df = pl.DataFrame({col: pl.Series(name=col, values=[], dtype=pl.Float64) 
    #                             for col in data1.columns})
    #    data2 = empty_df

    # 这时 data1, data2 都非 None
    all_cols = sorted(set(data1.columns) | set(data2.columns))
    print(all_cols)

    ## 先补齐缺失列
    #def ensure_cols(df: pl.DataFrame, all_cols: list[str]) -> pl.DataFrame:
    #    missing = set(all_cols) - set(df.columns)
    #    for m in missing:
    #        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(m))
    #    # 再统一列顺序
    #    return df.select(all_cols)

    #data1 = ensure_cols(data1, all_cols)
    #data2 = ensure_cols(data2, all_cols)

    # 最后再次统一类型为 Float64 (可以不加 strict=False, 看你需求)
# 假设 all_cols 包含 'sn'，我们在转换前将其排除
    cols_to_cast = [c for c in all_cols if c != 'sn']

    data1 = data1.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in cols_to_cast])
    data2 = data2.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in cols_to_cast])


    return pl.concat([data1, data2], how="vertical")


#def data_combined(data1: pl.DataFrame, data2: pl.DataFrame) -> pl.DataFrame:
#
#    # 先找出所有列并排序（这部分和你之前的逻辑一致）
#    all_cols = set(data1.columns).union(set(data2.columns))
#    all_cols_sorted = sorted(all_cols)
#
#    # 只选取并按顺序排列
#    pl_data1 = data1.select(all_cols_sorted)
#    pl_data2 = data2.select(all_cols_sorted)
#
#    # 将所有列统一转为 float64（strict=False 允许非 float 列保持原状）
#    pl_data1 = pl_data1.with_columns([pl.all().cast(pl.Float64, strict=False)])
#    pl_data2 = pl_data2.with_columns([pl.all().cast(pl.Float64, strict=False)])
#
#    combined_monthly_df = pl.concat([pl_data1, pl_data2], how="vertical")
#    return combined_monthly_df

#
def get_train_test_split(df,test_size):
    #X = df.drop(['label','sn'], axis=1)  # 特征
    X = df.drop(['label','sn'])  # 特征
    y = df['label']  # 标签
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,  stratify=y)
    return X_train, X_test, y_train, y_test
    
def get_train_test_split1(df,test_size):
    #X = df.drop(['label','sn'], axis=1)  # 特征
    
    X1 = pl.read_csv('train_data.csv')

# 打乱数据行（可选：设置种子以确保可重复性）
     # 使用 Polars 进行打乱
    float_columns = [col for col in X1.columns if col not in ['sn', 'createtime']]
    x2=pl.read_csv('val_data.csv')
    
    X1 = X1.with_columns([
            pl.col(col).cast(pl.Float64) for col in float_columns
        ])
    x2 = x2.with_columns([
            pl.col(col).cast(pl.Float64) for col in float_columns
        ])

    X_train=X1.drop(['label','sn'])  # 特征
    y_train=X1['label'] 
    X_test=x2.drop(['label','sn'])
    y_test= x2['label'] 
    #return X_train, X_test, y_train, y_test

def preprocess1(data, days=0, outdays=30):
    df, columns  = DataProcessor(data,None).main()

    numeric_cols = [
    col for col, dtype in zip(df.columns, df.dtypes)
    if dtype in pl.NUMERIC_DTYPES and col not in ('sn', 'createtime')]

    df = df.with_columns(
            pl.col('createtime').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
        )
    
    df = df.with_columns(pl.col('createtime').cast(pl.Datetime).dt.date().alias('date'))
    result = df.group_by(['sn', 'date']).agg(pl.col('createtime').n_unique()).sort(by=['sn', 'date'])

    sn_date_last_day = result.filter(pl.col('createtime') > 100).group_by('sn').agg(pl.col('date').max())
    not_melted_lastday = df.join(sn_date_last_day, on=['sn','date'], how='inner')
    return not_melted_lastday