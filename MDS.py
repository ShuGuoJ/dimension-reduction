import numpy as np


def mds(data, d):
    # 归一化
    # data:[features, n_samples]
    data -= np.min(data, axis=1, keepdims=True)
    data /= np.max(data, axis=1, keepdims=True)

    # 计算距离矩阵
    dist = np.transpose(data)@data
    B = dist - np.mean(dist, axis=1, keepdims=True) - np.mean(dist, axis=0, keepdims=True) - np.mean(dist)

    #特征分解
    eigenvalue, eigenvector = np.linalg.eig(B)
    index = np.argsort(eigenvalue)[::-1][:d].tolist()
    d_eigenvalue = eigenvalue[index]
    d_eigenvector = eigenvector[:,index]
    d_eigenvalue_matrix = np.eye(d) * d_eigenvalue
    return np.sqrt(d_eigenvalue_matrix)@np.transpose(d_eigenvector)

from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
print(np.sqrt(np.sum(np.square(data[0] - data[1]))))
new_data = np.transpose(mds(np.transpose(data), 2))
print(np.sqrt(np.sum(np.square(new_data[0] - new_data[1]))))
from matplotlib import pyplot as plt
plt.scatter(new_data[:,0], new_data[:,1],c=iris.target)
plt.savefig('mds.jpg')