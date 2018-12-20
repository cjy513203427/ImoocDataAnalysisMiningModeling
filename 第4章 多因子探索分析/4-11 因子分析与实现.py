import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# 防止pandas输出结果省略
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

df = pd.read_csv("./data/hr.csv")
df = df.dropna(axis=0, how="any")
df = df[df["last_evaluation"] <= 1][df["salary"] != "nme"][df["department"] != "sale"]

from sklearn.decomposition import PCA
from sklearn import preprocessing
my_pca = PCA(n_components=7)
df = df.drop(labels=["salary","department","left"],axis=1)
lower_mat = my_pca.fit_transform(df)

print("Ratio:\n",my_pca.explained_variance_ratio_)
data_scaled = pd.DataFrame(preprocessing.scale(df),columns = df.columns)
print("Column\n",data_scaled)
sns.heatmap(pd.DataFrame(lower_mat).corr(),vmin=-1,vmax=1,cmap=sns.color_palette("RdBu",n_colors=128))
plt.show()