'''
Author: franco
Date: 2024-11-26 14:48:47
LastEditors: franco
LastEditTime: 2024-11-26 14:49:04
Description: 
'''
from functools import  wraps
import time
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