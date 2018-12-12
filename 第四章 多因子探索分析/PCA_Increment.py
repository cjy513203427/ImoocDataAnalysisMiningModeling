import numpy as np


def PCA(data, n_components=100000000):
    mean_values = np.mean(data, axis=0)
    mid = data - mean_values
    # 计算协方差，rowvar=False表示对列进行计算
    cov_mat = np.cov(mid, rowvar=False)
    from scipy import linalg
    # 求特征值和特征向量
    eig_vals, eig_vects = linalg.eig(np.mat(cov_mat))
    eig_val_index = np.argsort(eig_vals)
    eig_val_index = eig_val_index[:-(n_components + 1):-1]
    eig_vects = eig_vects[:, eig_val_index]
    low_dim_mat = np.dot(mid, eig_vects)
    return low_dim_mat, eig_vals

data = np.array([np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1]),
            np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])]).T
print(PCA(data,n_components=1))
