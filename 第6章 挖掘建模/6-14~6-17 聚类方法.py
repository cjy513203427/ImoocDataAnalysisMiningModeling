import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

n_samples = 1000
circles = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
moons = make_moons(n_samples=n_samples, noise=0.05)
blobs = make_blobs(n_samples=n_samples, random_state=8, center_box=(-1, 1), cluster_std=0.1)
random_data = np.random.rand(n_samples, 2), None
# 输出样本点+标注
print(circles)

colors = "cmykbgr"
# 定义circle,moon,blob,random_data的数据
data = [circles, moons, blobs, random_data]
# 防止plt中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 不聚类、Kmeans聚类、DBSCAN聚类、层次聚类
models = [("没有聚类", None), ("Kmeans聚类", KMeans(n_clusters=3)),
          ("密度聚类", DBSCAN(min_samples=3, eps=0.2)),
          ("层次聚类", AgglomerativeClustering(n_clusters=3, linkage="ward"))]
f = plt.figure()
for inx, clt in enumerate(models):
    clt_name, clt_entity = clt
    for i, dataset in enumerate(data):
        X, Y = dataset
        if not clt_entity:
            clt_res = [0 for item in range(len(X))]
        else:
            clt_entity.fit(X)
            clt_res = clt_entity.labels_.astype(np.int)
        f.add_subplot(len(models), len(data), inx * len(data) + i + 1)
        plt.title(clt_name)
        [plt.scatter(X[p, 0], X[p, 1], color=colors[clt_res[p]]) for p in range(len(X))]
plt.show()
