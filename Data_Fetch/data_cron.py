from sqlalchemy import create_engine, Column, String, DateTime, func,Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import polars as pl
import pandas as pd

# 连接数据库的 URL
url = ""

# 创建 SQLAlchemy 引擎
engine = create_engine(
    url,
    pool_size=50,  # 连接池中的最大连接数
    max_overflow=5,  # 超出连接池后的最大连接数
    pool_timeout=30,  # 等待连接的超时时间（秒）
    pool_recycle=1800,  # 连接空闲1800秒后重置连接
    pool_pre_ping=True,  # 获取连接时，检测连接是否可用
)

# 创建基础类
Base = declarative_base()

# 定义 InverterData 模型类（与表 ts_bd_ods.ods_power_inverter_data_di 映射）
class InverterData(Base):
    __tablename__ = "ods_power_inverter_data_di"
    __table_args__ = {"schema": "ts_bd_ods"}
    invertersn = Column(String, primary_key=True)  # 使用 'sn' 作为主键
    temperature = Column(Float)
    vpv1 = Column(Float)
    vpv2 = Column(Float)
    vpv3 = Column(Float)
    vpv4 = Column(Float)
    vpv5 = Column(Float)
    vpv6 = Column(Float)
    vpv7 = Column(Float)
    vpv8 = Column(Float)
    vpv9 = Column(Float)
    vpv10 = Column(Float)
    vpv11 = Column(Float)
    vpv12 = Column(Float)
    vpv13 = Column(Float)
    vpv14 = Column(Float)
    vpv15 = Column(Float)
    vpv16 = Column(Float)
    vpv17 = Column(Float)
    vpv18 = Column(Float)
    vpv19 = Column(Float)
    vpv20 = Column(Float)
    vpv21 = Column(Float)
    vpv22 = Column(Float)
    vpv23 = Column(Float)
    vpv24 = Column(Float)
    vpv25 = Column(Float)
    vpv26 = Column(Float)
    vpv27 = Column(Float)
    vpv28 = Column(Float)
    vpv29 = Column(Float)
    vpv30 = Column(Float)
    ipv1 = Column(Float)
    ipv2 = Column(Float)
    ipv3 = Column(Float)
    ipv4 = Column(Float)
    ipv5 = Column(Float)
    ipv6 = Column(Float)
    ipv7 = Column(Float)
    ipv8 = Column(Float)
    ipv9 = Column(Float)
    ipv10 = Column(Float)
    ipv11 = Column(Float)
    ipv12 = Column(Float)
    ipv13 = Column(Float)
    ipv14 = Column(Float)
    ipv15 = Column(Float)
    ipv16 = Column(Float)
    ipv17 = Column(Float)
    ipv18 = Column(Float)
    ipv19 = Column(Float)
    ipv20 = Column(Float)
    ipv21 = Column(Float)
    ipv22 = Column(Float)
    ipv23 = Column(Float)
    ipv24 = Column(Float)
    ipv25 = Column(Float)
    ipv26 = Column(Float)
    ipv27 = Column(Float)
    ipv28 = Column(Float)
    ipv29 = Column(Float)
    ipv30 = Column(Float)
    vac1 = Column(Float)
    vac2 = Column(Float)
    vac3 = Column(Float)
    iac1 = Column(Float)
    iac2 = Column(Float)
    iac3 = Column(Float)
    fac1 = Column(Float)
    fac2 = Column(Float)
    fac3=Column(Float)
    switchstatus=Column(Float)
    powercurvenumber=Column(String)
    errorcode=Column(String)
    alertcode=Column(String)
    alertcodelist=Column(String)
    errorcodelist=Column(String)
    pac = Column(Float)
    eday = Column(Float)
    etotal = Column(Float)
    ttotal = Column(Float)
    outputpowerratio = Column(Float)
    activepower = Column(Float)
    reactivepower = Column(Float)
    inspectingpower = Column(Float)
    powerfactor = Column(Float)
    createtime = Column(DateTime)
    sn = Column(String)  # 保持 'sn' 字段
    ds=Column(DateTime)
    # 你可以根据实际情况添加更多字段

# 创建 Session 类
Session = sessionmaker(bind=engine)
session = Session()

def query_data_and_save_to_parquet(start_date: str, end_date: str, output_dir: str):
    # 将 start_date 和 end_date 转换为 datetime 类型
    current_date = datetime.strptime(start_date, '%Y%m%d')

    # 循环遍历每一天的数据
    while current_date <= datetime.strptime(end_date, '%Y%m%d'):
        # 格式化当前日期为字符串
        formatted_date = current_date.strftime('%Y%m%d')  # 使用 YYYY-MM-DD 格式

        # 执行查询：筛选出 ds 在当前日期，且 sn 以 'TP' 开头的记录
        query = session.query(InverterData).filter(
            func.date(InverterData.ds) == formatted_date,  # 使用 func.date() 来提取日期部分进行比较
            InverterData.sn.like('TP%')  # 按 sn 字段筛选以 'TP' 开头的序列号
        )

        # 将查询结果转换为 polars DataFrame
        df = pl.from_pandas(pd.read_sql(query.statement, session.bind))

        # 如果有数据，保存为 parquet 格式
        if df.shape[0] > 0:
            parquet_filename = f"{output_dir}/ods_power_inverter_data_{formatted_date}.parquet"
            df.write_parquet(parquet_filename)
            print(f"Data for {formatted_date} saved to {parquet_filename}")
            print(f"共{len(df)}条数据！")
        else:
            print(f"No data found for {formatted_date}")
        
        # 增加一天，继续循环
        current_date += timedelta(days=1)
if __name__ == "__main__":
    # 定义时间区间和输出文件夹（根据实际需求调整）

    output_dir="/data/inverter_power_data/data_di/"

    yesterday = datetime.today() - timedelta(days=1)

# 格式化为 YYYYMMDD
    yesterday_str = yesterday.strftime('%Y%m%d')
    query_data_and_save_to_parquet(yesterday_str, yesterday_str, output_dir)
