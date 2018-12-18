# encoding=utf-8
import pandas as pd
import numpy as np

df = pd.read_csv("./data/HR_.csv")

s_s = df["salary"]

print(s_s.value_counts())
#where的使用
# s_s.where(s_s=="nme")返回不等于nme的结果为NaN
# s_s.where(s_s=="nme")返回等于nme的结果为NaN
print("返回等于nme的结果为NaN\n", s_s.where(s_s != "nme"))

print("清空等于name的key和value", s_s.where(s_s != "nme").dropna())
