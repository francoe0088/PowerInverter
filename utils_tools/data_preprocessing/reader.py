from pathlib import Path
import polars as pl

def read_file(file_path):
    read_functions = {
        '.parquet': pl.read_parquet,
        '.csv': pl.read_csv,
    }
    ext = Path(file_path).suffix.lower()
    read_func = read_functions.get(ext)
    
    if not read_func:
        raise ValueError(f"不支持的文件格式: {ext}")
    return read_func(file_path)


