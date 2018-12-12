import numpy as np
import pandas as pd
import scipy.stats as ss

df = pd.DataFrame({"A":ss.norm.rvs(size = 10),"B":ss.norm.rvs(size = 10),"C":ss.norm.rvs(size = 10),
                   "D":np.random.randint(low=0,high=2,size=10)})
print(df)

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

X = df.loc[:,["A","B","C"]]
Y = df.loc[:,"D"]

from sklearn.feature_selection import SelectKBest,RFE,SelectFromModel
#Select features according to the k highest scores.
skb = SelectKBest(k=2)
#拟合结果
print("SelectKBest拟合结果\n",skb.fit(X,Y))
#变形结果
print("SelectKBest变形\n",skb.transform(X))
#RFE是recursive feature elimination回归特征消除，让回归特征消除过程中只保留no_features个最重要的特征，可以避免过度拟合
rfe = RFE(estimator=SVR(kernel="linear"),n_features_to_select=2,step=1)
print("RFE拟合结果\n",rfe.fit_transform(X,Y))

sfm = SelectFromModel(estimator=DecisionTreeRegressor(),threshold=0.001)
print("sfm拟合结果\n",sfm.fit_transform(X,Y))