# encoding=utf-8
import pandas as pd
import numpy as np

df = pd.read_csv("./data/HR_.csv")

sl_s = df["satisfaction_level"]
# 查看有没有异常值
print("查看有没有异常值\n", sl_s.isnull())

print("筛选出异常值\n", sl_s[sl_s.isnull()])

print("查看异常值详细信息\n", df[sl_s.isnull()])
# 丢弃空值行
sl_s = sl_s.dropna()
# 均值
print("均值\n", sl_s.mean())
# 标准差
print("标准差\n", sl_s.std())
# 最大值
print("最大值\n", sl_s.max())
# 中位数
print("中位数\n", sl_s.median())
# 四分位数
print("下四分位数\n", sl_s.quantile(q=0.25))

print("上四分位数\n", sl_s.quantile(q=0.75))
#偏度系数
print("偏度系数\n", sl_s.skew())
#峰度系数
print("峰度系数\n", sl_s.kurt())

print("获得该分布的数值\n", np.histogram(sl_s.values,bins=np.arange(0.0,1.1,0.1)))