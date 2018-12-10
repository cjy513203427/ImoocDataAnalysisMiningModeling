# encoding=utf-8
import pandas as pd
import numpy as np

# 防止pandas输出结果省略
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

df = pd.read_csv("./data/HR_.csv")
df = df.dropna(axis=0, how="any")
df = df[df["last_evaluation"] <= 1][df["salary"] != "nme"][df["department"] != "sale"]

import seaborn as sns
import matplotlib.pyplot as plt
#箱形图概念参照图3-24
#saturation:圈定方框的边界
#whis:上分位数取多少能到达上界
sns.boxplot(x=df["time_spend_company"],saturation=0.75,whis=3)
plt.show()
