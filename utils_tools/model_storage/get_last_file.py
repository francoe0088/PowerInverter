from pathlib import Path
from utils_tools.data_preprocessing.load_yaml import predict_config
def get_latest_file():
    test_file,seven_model_path,fifiteen_model_path=predict_config()
    #directory= "./output/saved_models/7_30/"
    directory=seven_model_path
    file_extension=".joblib"
    p = Path(directory)
    if not p.is_dir():
        #print(f"指定的目录不存在: {directory}")
        return None

    if file_extension:
        files = list(p.glob(f"*{file_extension}"))
    else:
        files = list(p.iterdir())

    if not files:
        #print("没有找到匹配的文件。")
        return None

    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    return latest_file

# 使用示例

