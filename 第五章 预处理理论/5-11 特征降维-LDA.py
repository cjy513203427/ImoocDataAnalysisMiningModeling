import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
Y = np.array([1,1,1,2,2,2])

print("特征降维-LDA\n",LinearDiscriminantAnalysis(n_components=1).fit_transform(X,Y))

clf = LinearDiscriminantAnalysis(n_components=1).fit(X,Y)
#Predict class labels for samples in X
print(clf.predict([[0.8,1]]))