import polars as pl
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from tqdm import tqdm  # 导入 tqdm
import re
from datetime import datetime

# 定义输入和输出路径
input_path = "/home/franco/power/inverter/new_code/inverter_dataset/shang_neng/data_di/"    # 原始 Parquet 文件的目录
output_path = "/home/franco/power/inverter1/inverter/code_v3/data_predict1/"  # 输出文件保存的目录

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# 遍历所有子目录，收集所有 Parquet 文件路径
def filter_date(start_date, end_date, parquet_files):
    """
    根据日期范围过滤 .parquet 文件。

    参数:
    start_date (int): 起始日期，格式为 YYYYMMDD，例如 20241105
    end_date (int): 结束日期，格式为 YYYYMMDD，例如 20241109
    parquet_files (list): .parquet 文件路径列表

    返回:
    list: 符合日期范围的 .parquet 文件路径列表
    """
    # 将整数类型的 start_date 和 end_date 转换为 datetime 对象
    start_date = datetime.strptime(str(start_date), '%Y%m%d')
    end_date = datetime.strptime(str(end_date), '%Y%m%d')

    # 用于存储符合条件的文件
    filtered_files = []

    # 遍历文件列表
    for file_path in parquet_files:
        # 从文件名中提取日期部分（假设格式为 YYYYMMDD）
        file_name = os.path.basename(file_path)
        match = re.search(r'(\d{8})', file_name)
        if match:
            # 提取并转换为 datetime 对象
            file_date = datetime.strptime(match.group(1), '%Y%m%d')

            # 根据文件日期过滤
            if start_date <= file_date <= end_date:
                filtered_files.append(file_path)  # 添加符合条件的文件到列表


    return filtered_files

def collect_parquet_files(directory):
    parquet_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".parquet"):
                parquet_files.append(os.path.join(root, f))
    return parquet_files


# 读取单个Parquet文件的唯一sn值
def get_unique_sns(file_path):
    try:
        df = pl.read_parquet(file_path, columns=["sn"])
        unique_sns = df["sn"].unique().to_list()
        print(f"文件 {os.path.basename(file_path)} 中找到 {len(unique_sns)} 个唯一的 sn。")
        return unique_sns
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return []

# 并行收集所有唯一的sn
def collect_all_unique_sns(files, max_workers=4):
    unique_sns_set = set()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_unique_sns, file): file for file in files}
        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="收集唯一 sn"):
            sn_list = future.result()
            unique_sns_set.update(sn_list)
    return list(unique_sns_set)


# 将sn分批，每批1000个
def batch_sns(sns, batch_size=1000):
    total = len(sns)
    num_batches = math.ceil(total / batch_size)
    for i in range(num_batches):
        yield sns[i*batch_size : (i+1)*batch_size]



# 处理单个批次的函数
def process_batch(batch_sn, batch_index, files, output_directory):
    try:
        batch_df_list = []
        for file_path in files:
            try:
                df = pl.read_parquet(file_path)
                # 筛选当前批次的sn
                df_filtered = df.filter(pl.col("sn").is_in(batch_sn))
                if not df_filtered.is_empty():
                    batch_df_list.append(df_filtered)
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
        if batch_df_list:
            # 合并所有数据
            combined_df = pl.concat(batch_df_list, rechunk=True)
            # 按 'sn' 和 'createtime' 排序
            sorted_df = combined_df.sort(["sn", "createtime"])
            # 定义输出文件路径
            output_file = os.path.join(output_directory, f"output_batch_{batch_index + 1}.parquet")
            # 写入Parquet文件
            sorted_df.write_parquet(output_file)
            print(sorted_df)
            print(f"已写入 {output_file}，包含 {sorted_df.shape[0]} 行。")
        else:
            print(f"批次 {batch_index + 1} 没有找到对应的数据。")
    except Exception as e:
        print(f"处理批次 {batch_index + 1} 时出错: {e}")

# 并行处理所有批次
def process_all_batches(batches, files, output_directory, max_workers=2):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_batch, batch, idx, files, output_directory): idx
            for idx, batch in enumerate(batches)
        }
        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理批次"):
            batch_index = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"批次 {batch_index + 1} 处理时出错: {e}")

# 启动批次处理



def main_inactive_data(start_date,end_date):
    try:
        start_date=20241011
        end_date=20241224
        parquet_files = collect_parquet_files(input_path)
        parquet_files=filter_date(start_date, end_date, parquet_files)
        all_unique_sns = collect_all_unique_sns(parquet_files, max_workers=8)
        print(f"总共有 {len(all_unique_sns)} 个唯一的 sn。")
        batches = list(batch_sns(all_unique_sns, batch_size=1000))
        print(f"总共有 {len(batches)} 个批次，每批 1000 个 sn（最后一批可能少于1000个）。")
        process_all_batches(batches, parquet_files, output_path, max_workers=2)
        print("所有批次的数据处理完成。")
        return "数据处理完成"
    except:
        return "数据处理失败"