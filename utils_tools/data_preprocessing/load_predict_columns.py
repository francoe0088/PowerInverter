import pickle
def load_columns(filename):
        # 从文件加载 ipv_columns 和 vpv_columns
    with open(filename, 'rb') as f:
        columns_data = pickle.load(f)
        all_columns = columns_data['all_columns']

    return all_columns
        