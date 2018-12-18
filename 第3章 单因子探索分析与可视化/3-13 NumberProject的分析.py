# encoding=utf-8
import pandas as pd
import numpy as np

df = pd.read_csv("./data/HR_.csv")

np_s = df["number_project"]

print(np_s[np_s.isnull()])

print("平均值\n", np_s.mean())

print("标准差\n", np_s.std())

print("中位数\n", np_s.median())

print("最大值\n", np_s.max())

print("最小值\n", np_s.min())

print("偏态系数\n", np_s.skew())

print("峰态系数\n", np_s.kurt())

print("数值构成比例\n", np_s.value_counts())

print("数值构成比例规范化\n", np_s.value_counts(normalize=True))

print("按索引排序数值构成比例规范化\n", np_s.value_counts(normalize=True).sort_index())