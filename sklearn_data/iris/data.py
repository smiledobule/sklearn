"""
https://zhuanlan.zhihu.com/p/31785188
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import _rebuild
from matplotlib.colors import ListedColormap

_rebuild()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

iris = datasets.load_iris()
print(type(iris))

# 特征名
print(iris.feature_names)
# 标签名
print(iris.target_names)

# 特征
# print(iris.data)

# 标签
# print(iris.target)

print(iris.data.shape)
print(iris.target.shape)

# store features matrix in X
X = iris.data

# store features matrix in y
y = iris.target

X_sepal = X[:, :2]
X_petal = X[:, 2:4]

# 绘制萼片和花瓣长度、宽度散点图
# plt.subplot(211)
# plt.scatter(X_sepal[:, 0], X_sepal[:, 1], c=y, cmap=plt.cm.gnuplot)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.subplot(212)
# plt.scatter(X_petal[:, 0], X_petal[:, 1], c=y, cmap=plt.cm.gnuplot)
# plt.xlabel('Petal length')
# plt.ylabel('Petal width')
# plt.show()

# 查看每个特征的分布
# x = np.arange(len(y))
# plt.figure(figsize=(8, 8))
# plt.subplot(221)
# plt.hist(X_sepal[:, 0], bins=10)
# plt.xlabel('Sepal length')
# plt.ylabel('Label number')
# plt.subplot(222)
# plt.hist(X_sepal[:, 1], bins=10)
# plt.xlabel('Sepal width')
# plt.ylabel('Label number')
# plt.subplot(223)
# plt.hist(X_petal[:, 0], bins=10)
# plt.xlabel('petal length')
# plt.ylabel('Label number')
# plt.subplot(224)
# plt.hist(X_petal[:, 1], bins=10)
# plt.xlabel('petal length')
# plt.ylabel('Label number')
# plt.show()

# 皮尔逊相关系数
print(pearsonr(X_sepal[:, 0], y))
print(pearsonr(X_sepal[:, 1], y))
print(pearsonr(X_petal[:, 0], y))
print(pearsonr(X_petal[:, 1], y))

# 相关系数矩阵
# X_df = pd.DataFrame(X)
# X_df['label'] = y
# pearson_corr = X_df.corr(method='pearson')
# plt.figure()
# plt.subplot()
# sns.heatmap(pearson_corr, cmap='Blues', annot=True)
# plt.title('Pearson')
#
# kendall_corr = X_df.corr(method='kendall')
# plt.figure()
# plt.subplot()
# sns.heatmap(kendall_corr, cmap='Blues', annot=True)
# plt.title('Kendall')
#
# spearman_corr = X_df.corr(method='spearman')
# plt.figure()
# plt.subplot()
# sns.heatmap(spearman_corr, cmap='Blues', annot=True)
# plt.title('Spearman')
# plt.show()

# PCA降维 用于展示三维数据分布
# X_reduced = PCA(n_components=3).fit_transform(iris.data)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_title('Iris Dataset by PCA')
# ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y)
# ax.set_xlabel('First eigenvector')
# ax.set_ylabel('Second eigenvector')
# ax.set_zlabel('Third egienvector')
# plt.show()

# 训练集 测试集
rand_index = np.random.permutation(len(X))
x_train = X[rand_index[:-10]]
y_train = y[rand_index[:-10]]
x_test = X[rand_index[-10:]]
y_test = y[rand_index[-10:]]


# KNN
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(y_pred)
print(y_test)
matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
print(matrix)
print(accuracy_score(y_test, y_pred))
# 决策边界
x_sepal = iris.data[:, :2]

x_min, x_max = x_sepal[:, 0].min() - 0.5, x_sepal[:, 0].max() + 0.5
y_min, y_max = x_sepal[:, 1].min() - 0.5, x_sepal[:, 1].max() + 0.5
cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
h = 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
knn.fit(x_sepal, y)
z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, z, cmap=cmap_light)
plt.scatter(x_sepal[:, 0], x_sepal[:, 1], c=y)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()



