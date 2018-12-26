import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
# 第一部分是data，有四个属性；第二个部分是target，代表它的三种分类
print(iris)
labels = np.copy(iris.target)
# 通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)
random_unlabeled_points = np.random.rand(len(iris.target))
random_unlabeled_points = random_unlabeled_points < 0.3
# print(np.random.rand(len(iris.target)))
Y = labels[random_unlabeled_points]
labels[random_unlabeled_points] = -1
# print(iris.target)
# print(labels)
print("Unlabeled Number:", list(labels).count(-1))

from sklearn.semi_supervised import LabelPropagation

label_prop_model = LabelPropagation()
label_prop_model.fit(iris.data, labels)
Y_pred = label_prop_model.predict(iris.data)
Y_pred = Y_pred[random_unlabeled_points]
from sklearn.metrics import accuracy_score, recall_score, f1_score

print("ACC:", accuracy_score(Y, Y_pred))
print("REC:", recall_score(Y, Y_pred,average="micro"))
print("F-Score:", f1_score(Y, Y_pred,average="micro"))
