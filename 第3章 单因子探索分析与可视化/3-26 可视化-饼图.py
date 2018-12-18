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

# 普通版饼图
lbs = df["department"].value_counts().index
plt.pie(df["department"].value_counts(normalize=True), labels=lbs, autopct="%1.1f%%")
plt.show()
# 暴露某因子，没弄懂这种写法
explodes = [0.1 if i == "sales" else 0 for i in lbs]
# autopct显示数字
# explodes暴露因子
plt.pie(df["department"].value_counts(normalize=True), explode=explodes, labels=lbs, autopct="%1.1f%%",
        colors=sns.color_palette("Reds"))
plt.show()
