# encoding=utf-8
import pandas as pd
import numpy as np

df = pd.read_csv("./data/HR_.csv")

wa_s = df["Work_accident"]

print("value_counts统计\n", wa_s.value_counts().sort_index())
#1出事故0未出事故
print("事故概率\n",wa_s.mean())