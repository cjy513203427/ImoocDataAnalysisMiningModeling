import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/hr.csv")
df = df.dropna(axis=0, how="any")
df = df[df["last_evaluation"] <= 1][df["salary"] != "nme"][df["department"] != "sale"]
# 画相关图
sns.set_context(font_scale=1.5)
# 泛红的负相关，泛蓝的正相关
sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap=sns.color_palette("RdBu", n_colors=128))
# plt.show()

s1 = pd.Series(["X1", "X1", "X2", "X2", "X2", "X2"])
s2 = pd.Series(["Y1", "Y1", "Y1", "Y2", "Y2", "Y2"])


# 计算熵
def getEntropy(s):
    # 将s转换成Series
    if not isinstance(s, pd.core.series.Series):
        s = pd.Series(s)
    prt_ary = pd.groupby(s, by=s).count().values / float(len(s))
    # return熵
    return -(np.log2(prt_ary) * prt_ary).sum()


print("Entroy:", getEntropy(s2))


# 计算条件熵
# 当X=X1时，有Y1,Y1;当X=X2时，有Y1,Y2,Y2,Y2
def getCondEntropy(s1, s2):
    d = dict()
    for i in list(range(len(s1))):
        d[s1[i]] = d.get(s1[i], []) + [s2[i]]
    return sum([getEntropy(d[k]) * len(d[k]) / float(len(s1)) for k in d])


print("CondEntropy", getCondEntropy(s1, s2))


# 计算熵增益
def getEntropyGain(s1, s2):
    return getEntropy(s2) - getCondEntropy(s1, s2)


print("EntropyGain", getEntropyGain(s1, s2))


# 计算熵增益率
def getEntropyGainRatio(s1, s2):
    return getEntropyGain(s1, s2) / getEntropy(s2)


print("EntropyGainRatio", getEntropyGainRatio(s2, s1))

# 衡量离散值相关性
import math


def getDiskreteCorr(s1, s2):
    return getEntropyGain(s1, s2) / math.sqrt(getEntropy(s1) * getEntropy(s2))


print("DiskreteCorr", getDiskreteCorr(s1, s2))


# 求概率平方和
def getProbSS(s):
    # 将s转换成Series
    if not isinstance(s, pd.core.series.Series):
        s = pd.Series(s)
    prt_ary = pd.groupby(s, by=s).count().values / float(len(s))
    # return熵
    return sum(prt_ary ** 2)


# 求Gini系数
def getGini(s1, s2):
    d = dict()
    for i in list(range(len(s1))):
        d[s1[i]] = d.get(s1[i], []) + [s2[i]]
    return 1 - sum([getProbSS(d[k]) * len(d[k]) / float(len(s1)) for k in d])


print("Gini系数\n", getGini(s1, s2))
