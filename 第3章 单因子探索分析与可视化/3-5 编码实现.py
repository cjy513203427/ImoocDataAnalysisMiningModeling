import pandas as pd

df = pd.read_csv('./data/hr.csv')

# dataframe的类型
print(type(df))

print(type(df['satisfaction_level']))

# Series可以直接求中位数、均值
print("satisfaction_level列的均值\n", df['satisfaction_level'].mean())
# 中位数
print("satisfaction_level列的中位数\n", df['satisfaction_level'].median())
# 算df的四分位数
print("算df的四分位数\n", df.quantile(q=0.25))

print("算df satisfaction_level列的四分位数\n", df['satisfaction_level'].quantile(q=0.25))
# 众数
print("算df的众数\n", df.mode())

print("算df satisfaction_level列的众数\n", df['satisfaction_level'].mode())
# 标准差
print("算df satisfaction_level列的标准差\n", df['satisfaction_level'].std())
# 方差
print("算df satisfaction_level列的方差\n", df['satisfaction_level'].var())

print("算df satisfaction_level列的和\n", df['satisfaction_level'].sum())
# 偏态系数
# 结果为负说明是负偏，大多数值大于平均数，是大多数人比较满意的状态
print("算df satisfaction_level列的偏态系数\n", df['satisfaction_level'].skew())
# 峰态系数
# 结果为-0.67，说明该分布要平缓些
print("算df satisfaction_level列的峰态系数\n", df['satisfaction_level'].kurt())

import scipy.stats as ss

# 正态分布
# m:均值;v:方差;s:偏态系数;k:峰态系数;
print(ss.norm.stats(moments="mvsk"))
# pdf计算该点的纵坐标值，是0.398
print("标准正态分布在0的数\n", ss.norm.pdf(0.0))
# ppf输入值必须是0到1之间，表示返回结果当概率积分为0.9时，是-∞到1.2815515655446004
print("返回结果当概率积分为0.9时，是-∞到到多少\n", ss.norm.ppf(0.9))
# cdf表示从-∞到2的累计概率是0.9772
print("表示从-∞到2的累计概率是\n", ss.norm.cdf(2))
# 验证正态分布-2σ到＋2σ的累积概率
print("标准正态分布-2σ到＋2σ的累积概率\n", ss.norm.cdf(2) - ss.norm.cdf(-2))
#得到十个符合正态分布的数字
print("符合正态分布的数字\n",ss.norm.rvs(size=10))
#卡方分布、t分布和f分布，操作与正态分布类似
ss.chi2
ss.t
ss.f
#抽样十个
print ("抽样十个\n",df.sample(n=10))
#抽样0.001
print ("抽样0.001\n",df.sample(frac=0.001))
#在satisfaction_level列抽十条数据
print (df["satisfaction_level"].sample(n=10))