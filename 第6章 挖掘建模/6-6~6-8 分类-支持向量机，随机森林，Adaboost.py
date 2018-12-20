import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

# 防止pandas输出结果省略
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

import os
import pydotplus

os.environ["PATH"] += os.pathsep + "D:/Graphviz/bin/"


# sl:satisfaction_level---False:MinMaxScaler;True:StandardScaler
# le:last_evaluation---False:MinMaxScaler;True:StandardScaler
# npr:number_project---False:MinMaxScaler;True:StandardScaler
# tsc:time_spend_company---False:MinMaxScaler;True:StandardScaler
# wa:Work_accident---False:MinMaxScaler;True:StandardScaler
# pl5:promotion_last_5years---False:MinMaxScaler;True:StandardScaler
# dp:department---False:LabelEncoding;True:OneHotEncoding
# slr:salary---False:LabelEncoding;True:OneHotEncoding
# ldn:降维---False:不降维;True:降维
# ld_n:Dimension
def hr_preprocessing(sl=False, le=False, npr=False, tsc=False, wa=False, p15=False, dp=False, slr=False, lower_d=False,
                     ld_n=1):
    df = pd.read_csv("./data/HR_.csv")
    # 1.清洗数据
    df = df.dropna(subset=["satisfaction_level", "last_evaluation"])
    df = df[df["satisfaction_level"] <= 1][df["salary"] != "nme"]
    # 2.得到标注
    label = df["left"]
    # 把left列删掉
    df = df.drop("left", axis=1)
    # 3.特征选择
    # 4.特征处理
    scaler_list = [sl, le, npr, tsc, wa, p15]
    column_list = ["satisfaction_level", "last_evaluation", "number_project", "time_spend_company", "Work_accident",
                   "promotion_last_5years"]
    # 经过处理satisfaction_level列数据都变小
    for i in range(len(scaler_list)):
        if not scaler_list[i]:
            df[column_list[i]] = \
                MinMaxScaler().fit_transform(df[column_list[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            df[column_list[i]] = \
                StandardScaler().fit_transform(df[column_list[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
    scaler_list = [slr, dp]
    column_list = ["salary", "department"]
    for i in range(len(scaler_list)):
        if not scaler_list[i]:
            if column_list[i] == "salary":
                df[column_list[i]] = [map_salary(s) for s in df["salary"].values]
            else:
                # 处理department
                df[column_list[i]] = LabelEncoder().fit_transform(df[column_list[i]])
        else:
            # 独热化
            df = pd.get_dummies(df, columns=[column_list[i]])
    if lower_d:
        return PCA(n_components=ld_n).fit_transform(df.values), label
    return df, label


d = dict([("low", 0), ("medium", 1), ("high", 2)])


def map_salary(s):
    # key不存在dict.keys()中时，返回0
    return d.get(s, 0)


def hr_modeling(features, label):
    from sklearn.model_selection import train_test_split
    f_v = features.values
    f_names = features.columns.values
    l_v = label.values
    X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
    X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=0.25)
    print(len(X_train), len(X_validation), len(X_test))

    # KNN
    from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
    from sklearn.metrics import accuracy_score, recall_score, f1_score
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.externals.six import StringIO
    from sklearn.tree import export_graphviz
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    models = []
    models.append(("KNN", KNeighborsClassifier(n_neighbors=3)))
    models.append(("GaussianNB", GaussianNB()))
    models.append(("BernoulliNB", BernoulliNB()))
    models.append(("DecisionTree", DecisionTreeClassifier()))
    models.append(("DecisionTreeEntropy", DecisionTreeClassifier(criterion="entropy")))
    #C是惩罚度，越高精准度越高，运算速度越慢
    models.append(("SVM Classifier",SVC(C=1000)))
    models.append(("OriginalRandomForest",RandomForestClassifier()))
    models.append(("RandomForest", RandomForestClassifier(n_estimators=1000,max_features=None)))
    models.append(("AdaBoost",AdaBoostClassifier(n_estimators=1000)))
    for clf_name, clf in models:
        clf.fit(X_train, Y_train)
        xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
        # 对代码6-2的优化
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = clf.predict(X_part)
            print(i)
            print(clf_name, "ACC:", accuracy_score(Y_part, Y_pred))
            print(clf_name, "REC:", recall_score(Y_part, Y_pred))
            print(clf_name, "F1", f1_score(Y_part, Y_pred))


def main():
    # print(hr_preprocessing(sl=True, le=True, dp=True, lower_d=True, ld_n=3))
    features, labels = hr_preprocessing()
    hr_modeling(features, labels)


# if __name__ == '__main__'的意思是：当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
# 当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
if __name__ == "__main__":
    main()
