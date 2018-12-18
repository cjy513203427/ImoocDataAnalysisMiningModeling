# encoding=utf-8
import pandas as pd
import numpy as np

# 防止pandas输出结果省略
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

df = pd.read_csv("./data/HR_.csv")
df = df.where(df != "nme").dropna()
df = df.dropna(axis=0, how="any")
df = df[df["last_evaluation"] <= 1][df["salary"] != "nme"][df["department"] != "sale"]

import seaborn as sns
import matplotlib.pyplot as plt
#seaborn设置样式
sns.set_style(style="whitegrid")
sns.set_context(context="poster",font_scale=0.5)
sns.set_palette("summer")
plt.title("SALARY")
plt.xlabel("salary")
plt.ylabel("Number")
#设置显示范围
#x为0到4，y取值为0到10000
plt.axis([0,4,0,10000])
#对x轴标注
plt.xticks(np.arange(len(df["salary"].value_counts()))+0.5,df["salary"].value_counts().index)
#绘制横纵坐标
plt.bar(np.arange(len(df["salary"].value_counts()))+0.5,df["salary"].value_counts(),width=0.5)
#标注纵坐标
for x,y in zip(np.arange(len(df["salary"].value_counts()))+0.5,df["salary"].value_counts()):
    plt.text(x,y,y,ha="center",va = "bottom")
plt.show()


#直接用seaborn绘制柱状图，简洁
sns.countplot(x="salary",data=df)
plt.show()
#多层绘制
sns.countplot(x="salary",hue="department",data=df)
plt.show()