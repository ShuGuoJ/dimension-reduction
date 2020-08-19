import numpy as np
import sklearn
from sklearn import datasets
from sklearn.decomposition import PCA
def dimension_reduction(data, d):
    # data:[features, n_samples]
    # 中心化
    data -= np.mean(data,axis=-1,keepdims=True)
    # 计算协方差矩阵
    covariance = data @ np.transpose(data) / (data.shape[1]-1)
    # 特征分解
    eigenvalue, eigenvector = np.linalg.eig(covariance)
    eigenvalue_top_d = np.argsort(eigenvalue)[::-1][:d].tolist()
    w = eigenvector[:,eigenvalue_top_d]
    variance = eigenvalue[eigenvalue_top_d]
    return np.transpose(w)@data, w

# iris = datasets.load_iris()
# data = iris.data
#
# new_data = np.transpose(dimension_reduction(np.transpose(data), 2))
# from matplotlib import pyplot as plt
# plt.scatter(new_data[:,0], new_data[:,1],c=iris.target)
# plt.savefig('pca.jpg')

# pca = PCA(n_components=2)
# pca.fit(data)
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# dimension_reduction(np.transpose(data), 2)





