# encoding=utf-8
import pandas as pd
import numpy as np

df = pd.read_csv("./data/HR_.csv")

d_s = df["department"]
#有sales和sale，可能是异常值，将其去掉
print(d_s.value_counts())

print("去掉sale\n",d_s.where(d_s!="sale").dropna())



