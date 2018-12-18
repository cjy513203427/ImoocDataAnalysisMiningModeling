import scipy.stats as ss
import numpy as np
import pandas as pd
from statsmodels.graphics.api import qqplot
from matplotlib import pyplot as plt

norm_dist = ss.norm.rvs(size=20)
print(norm_dist)

print("正态分布测试\n", ss.normaltest(norm_dist))

# 卡方检验
print("卡方检验\n", ss.chi2_contingency([[15, 95], [85, 5]]))
# t检验(双总体检验)
print("t检验\n", ss.ttest_ind(ss.norm.rvs(size=10), ss.norm.rvs(size=20)))
print("t检验大样本\n", ss.ttest_ind(ss.norm.rvs(size=100), ss.norm.rvs(size=200)))

# 方差检验
print("方差检验\n", ss.f_oneway([49, 50, 39, 40, 43], [28, 32, 30, 26, 24], [38, 40, 45, 42, 48]))

s1 = pd.Series([0.1,0.2,1.1,2.4,1.3,0.3,0.5])
s2 = pd.Series([0.5,0.4,1.2,2.5,1.1,0.7,0.1])
#相关系数
print("相关系数\n",s1.corr(s2))
print("spearman相关系数\n",s1.corr(s2,method="spearman"))

df = pd.DataFrame([s1,s2])
print(df.corr())
#reshape成m行n列
x = np.arange(10).astype(np.float).reshape(10,1)
print("x是\n",x)
#自己构造一个函数，用来检测线性回归的结果
y=x*3+4+np.random.random((10,1))
print("y是\n",y)

#线性回归
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg = reg.fit(x,y)
y_pred = reg.predict(x)
print("参数\n",reg.coef_)
print("截距\n",reg.intercept_)

data = np.array([np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1]),
            np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])]).T
print("data是\n",data)

from sklearn.decomposition import PCA
lower_dim = PCA(n_components=1)
lower_dim.fit(data)
#explained_variance_ratio_它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分
print("降维后结果\n",lower_dim.explained_variance_ratio_)

print("计算Transformed Data\n",lower_dim.transform(data))
