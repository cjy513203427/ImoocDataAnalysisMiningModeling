import pandas as pd

df = pd.DataFrame({"A": ["a0", "a1", "a1", "a2", "a3", "a4"], "B": ["b0", "b1", "b2", "b2", "b3", None],
                   "C": [1, 2, None, 3, 4, 5], "D": [0.1, 10.2, 11.4, 8.9, 9.1, 12], "E": [10, 19, 32, 25, 8, None],
                   "F": ["f0", "f1", "g2", "f3", "f4", "f5"]})
# 字符串为空是None，数字为空是NaN
print(df)

print("True是空值所在的位置\n", df.isnull())

print("去掉空值\n", df.dropna())

print("只删除B属性的空值\n", df.dropna(subset=["B"]))

print("显示A属性是否重复\n", df.duplicated(["A"]))

print("显示A属性和B属性是否重复\n", df.duplicated(["A", "B"]))
# keep可选参数,First表示保留第一行，Last表示保留最后一行，False表示不保留，全部删除
print("删除重复的值\n", df.drop_duplicates(["A"]))

print("空值标注成b*\n", df.fillna("b*"))

print("插值方法填充空值\n", df["E"].interpolate())
# F列只保留以f开头的数据
print(df[[True if item.startswith("f") else False for item in list(df["F"].values)]])
