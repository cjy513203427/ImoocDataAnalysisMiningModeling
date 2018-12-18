# encoding=utf-8
import pandas as pd
import numpy as np

df = pd.read_csv("./data/HR_.csv")

le_s = df["last_evaluation"]

print(le_s[le_s.isnull()])

print("平均值\n", le_s.mean())

print("标准差\n", le_s.std())

print("中位数\n", le_s.median())

print("最大值\n", le_s.max())

print("最小值\n", le_s.min())

print("偏态系数\n", le_s.skew())

print("峰态系数\n", le_s.kurt())

print("大于1的值\n", le_s[le_s > 1])
# 舍弃掉大于1的值
print("小于等于1的值\n", le_s[le_s <= 1])

le_s = df['last_evaluation']
q_low = le_s.quantile(q=0.25)
q_high = le_s.quantile(q=0.75)
q_interval = q_high - q_low
k = 1.5
# 筛选数据处于上四分位数-区间和下四分位数+区间之间的数
# q_low - k*q_interval < le_s < q_high + k*q_interval
print("小于等于1的值\n", le_s[le_s < q_high + k * q_interval][le_s > q_low - k * q_interval])

print("获得该分布的数值\n", np.histogram(le_s.values, bins=np.arange(0.0, 1.1, 0.1)))

le_s = le_s[le_s < q_high + k * q_interval][le_s > q_low - k * q_interval]
# 经过数据筛选后的各统计参数趋于正常
print("平均值\n", le_s.mean())

print("标准差\n", le_s.std())

print("中位数\n", le_s.median())

print("最大值\n", le_s.max())

print("最小值\n", le_s.min())

print("偏态系数\n", le_s.skew())

print("峰态系数\n", le_s.kurt())
