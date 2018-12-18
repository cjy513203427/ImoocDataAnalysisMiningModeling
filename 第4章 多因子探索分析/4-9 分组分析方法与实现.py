import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/hr.csv")
df = df.dropna(axis=0, how="any")
df = df[df["last_evaluation"] <= 1][df["salary"] != "nme"][df["department"] != "sale"]
#Draw a categorical scatterplot with non-overlapping points.
sns.barplot(x="salary",y="left",hue="department",data=df)
plt.show()
#该图片无法显示，原因未知
sl_s = df["satisfaction_level"]
sns.barplot(list(range(len(sl_s))),sl_s.sort_values())
plt.show()