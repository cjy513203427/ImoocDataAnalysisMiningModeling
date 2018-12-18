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

#seaborn设置样式
sns.set_style(style="whitegrid")
sns.set_context(context="poster",font_scale=0.5)
sns.set_palette("summer")

f = plt.figure()
#subplot(numRows, numCols, plotNum)
#图表的整个绘图区域被分成 numRows 行和 numCols 列
#然后按照从左到右，从上到下的顺序对每个子区域进行编号
f.add_subplot(1,3,1)
#画分布图
#kde=False表示没有分布曲线，hist=False表示没有直方图
sns.distplot(df["satisfaction_level"],bins=10)
f.add_subplot(1,3,2)
sns.distplot(df["last_evaluation"],bins=10)
f.add_subplot(1,3,3)
sns.distplot(df["average_monthly_hours"],bins=10)
plt.show()