# encoding=utf-8
import pandas as pd
import numpy as np

df = pd.read_csv("./data/HR_.csv")

amh_s = df["average_monthly_hours"]

print(amh_s[amh_s.isnull()])

print("平均值\n", amh_s.mean())

print("标准差\n", amh_s.std())

print("中位数\n", amh_s.median())

print("最大值\n", amh_s.max())

print("最小值\n", amh_s.min())

print("偏态系数\n", amh_s.skew())

print("峰态系数\n", amh_s.kurt())

amh_s = df['last_evaluation']
q_low = amh_s.quantile(q=0.25)
q_high = amh_s.quantile(q=0.75)
q_interval = q_high - q_low
k = 1.5
# 筛选数据处于上四分位数-区间和下四分位数+区间之间的数
# q_low - k*q_interval < amh_s < q_high + k*q_interval
amh_s = amh_s[amh_s < q_high + k * q_interval][amh_s > q_low - k * q_interval]

print("获得该分布的数值\n", np.histogram(amh_s.values, bins=10))
# np的value_counts左闭右开
print("获得该分布的数值\n", np.histogram(amh_s.values, bins=np.arange(amh_s.min(), amh_s.max() + 10, 10)))
# value_counts左开右闭
print("df的value_counts左开右闭", amh_s.value_counts(bins=np.arange(amh_s.min(), amh_s.max() + 10, 10)))
