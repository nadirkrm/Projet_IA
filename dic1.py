import pandas as pd

# 读取 CSV 文件
csv_file = 'output_unique_imdb_id.csv'  # 请将此文件名替换为实际文件名
df = pd.read_csv(csv_file)

# 将数据转换为字典
dic_vec_compressed = df.set_index('imdb_id')[['collection_id', 'genre_id', 'runtime', 'release_year']].to_dict(orient='index')

# 打印或使用字典
# 打印字典中的前三个值
for i, key in enumerate(dic_vec_compressed):
    if i < 5:  # 只打印前三个键值对
        print(f"{key}: {dic_vec_compressed[key]}")
    else:
        break

