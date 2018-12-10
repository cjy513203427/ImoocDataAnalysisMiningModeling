# encoding=utf-8
import pandas as pd
import numpy as np

# 防止pandas输出结果省略
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

df = pd.read_csv("./data/HR_.csv")
# axis=0是删除行；axis=1是删除列；how="any"是只有有一列空就删除；how="all"是所有列都为空才删除
df = df.dropna(axis=0, how="any")
print("删除空的结果\n", df)

df = df[df["last_evaluation"] <= 1][df["salary"] != "nme"][df["department"] != "sale"]
print(df)
# groupby
print("groupby\n", df.groupby("department").mean())

# 只获取一列的groupby
print("只获取一列的groupby\n", df.loc[:, ["last_evaluation", "department"]].groupby("department").mean())
# 只获取average_monthly_hours列的groupby的极差
print("只获取average_monthly_hours列的groupby的极差\n",
      df.loc[:, ["average_monthly_hours", "department"]].groupby("department")["average_monthly_hours"]
      .apply(lambda x: x.max() - x.min()))
