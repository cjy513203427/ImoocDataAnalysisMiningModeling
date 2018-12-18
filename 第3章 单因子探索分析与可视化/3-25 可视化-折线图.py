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

sub_df = df.groupby("time_spend_company").mean()
sns.pointplot(sub_df.index,sub_df["left"])
plt.show()
#可以得到覆盖范围
sns.pointplot(x="time_spend_company",y="left",data=df)
plt.show()
