import pandas as pd

# 查看合并后的文件
df = pd.read_csv(r".\data\processed_data.csv")
print("Columns in the merged file:", df.columns)
print("First few rows of the merged file:")
print(df.head())
