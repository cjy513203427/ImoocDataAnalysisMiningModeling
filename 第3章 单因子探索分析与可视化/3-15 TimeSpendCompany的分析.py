# encoding=utf-8
import pandas as pd
import numpy as np

df = pd.read_csv("./data/HR_.csv")

tsc_s = df["time_spend_company"]

print("value_counts统计\n", tsc_s.value_counts().sort_index())

print("平均值\n", tsc_s.mean())