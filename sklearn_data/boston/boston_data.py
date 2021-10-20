"""
https://blog.csdn.net/mingxiaod/article/details/85938251
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score


boston = load_boston()
print(type(boston))

names = boston.feature_names
print(boston.feature_names)

x = boston.data
y = boston.target
# plt.scatter(np.arange(len(y)), y)
# plt.show()
print(type(x))
print(type(y))
print(x.shape)
print(y.shape)

# 异常值处理
value_idx = [i for i in np.arange(len(y)) if y[i] >= 40]
x = np.delete(x, value_idx, axis=0)
y = np.delete(y, value_idx, axis=0)

# 特征和目标值 散点图
# for i in np.arange(len(name)):
#     plt.subplot(4, 4, i + 1)
#     plt.scatter(x[:, i], y)
#     plt.title(name[i])
#     pass
# # plt.show()
#
# x_df = pd.DataFrame(x)
# x_df['target'] = y
#
# pearson_corr = x_df.corr()
# plt.figure()
# plt.subplot()
# sns.heatmap(pearson_corr, cmap='Blues', annot=True)
# plt.title('Pearson')
#
# kendall_corr = x_df.corr(method='kendall')
# plt.figure()
# plt.subplot()
# sns.heatmap(kendall_corr, cmap='Blues', annot=True)
# plt.title('Kendall')
#
# spearman_corr = x_df.corr(method='spearman')
# plt.figure()
# plt.subplot()
# sns.heatmap(spearman_corr, cmap='Blues', annot=True)
# plt.title('Spearman')
# plt.show()

# 选择特征
"""
2 INDUS -0.58
5 RM 0.63
10 PTRATIO -0.56
12 LSTAT -0.85
"""
filter_df = pd.DataFrame()
# feature_names = ['INDUS', 'RM', 'PTRATIO', 'LSTAT']
feature_names = ['RM', 'PTRATIO', 'LSTAT']
# feature_names = ['RM', 'LSTAT']
names = list(names)
for idx in np.arange(len(names)):
    if names[idx] in feature_names:
        name = names[idx]
        filter_df[names.index(names[idx])] = x[:, idx]
        pass
    pass
features = filter_df.values
x_train, x_test, y_train, y_test = train_test_split(features, y, random_state=0, test_size=0.2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# 归一化
min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
y_train = min_max_scaler.fit_transform(y_train.reshape(-1, 1))

x_test = min_max_scaler.fit_transform(x_test)
y_test = min_max_scaler.fit_transform(y_test.reshape(-1, 1))

# 归一会分布
x_reduced = PCA(n_components=2).fit_transform(x_train)
fig = plt.figure()
ax = Axes3D(fig)
ax.set_title('Iris Dataset by PCA')
ax.scatter(x_reduced[:, 0], x_reduced[:, 1], y_train)
ax.set_xlabel('First eigenvector')
ax.set_ylabel('Second eigenvector')
ax.set_zlabel('Third egienvector')
plt.show()

lr = LinearRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
score = r2_score(y_test, lr_pred)
print(score)
print(lr.intercept_)
print(lr.coef_)


