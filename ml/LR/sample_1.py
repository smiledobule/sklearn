# # 任务：是否患有糖尿病（二分类）
# 模型：LR
# 数据集：皮马人糖尿病数据集
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# 切分数据集
df = pd.read_csv("./data/diabetes.csv")
target = df.pop("Outcome")
data = df.values
X = data
Y = target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

# LR模型预测
lr = LogisticRegression()  # 初始化LogisticRegression
lr.fit(X_train, Y_train)  # 调用LogisticRegression中的fit函数训练模型参数
lr_pres = lr.predict(X_test)  # 使用训练好的模型lr对X_test进行预测
print('准确率：', accuracy_score(Y_test, lr_pres))
print('精确率：', precision_score(Y_test, lr_pres))
print('召回率：', recall_score(Y_test, lr_pres))

# 混淆矩阵热点图
labels = [0, 1]
cm = confusion_matrix(Y_test, lr_pres, labels)
sns.heatmap(cm, annot=True, annot_kws={'size': 20, 'weight': 'bold', 'color': 'blue'})
plt.rc('font', family='Arial Unicode MS', size=14)
plt.title('混淆矩阵', fontsize=20)
plt.xlabel('Actual', fontsize=14)
plt.ylabel('Predict', fontsize=14)
plt.show()

# ROC曲线和AUC
lr_pres_proba = lr.predict_proba(X_test)[::, 1]
fpr, tpr, thresholds = roc_curve(Y_test, lr_pres_proba)
auc = roc_auc_score(Y_test, lr_pres_proba)
plt.figure(figsize=(5, 3), dpi=100)
plt.plot(fpr, tpr, label="AUC={:.2f}".format(auc))
plt.legend(loc=4, fontsize=10)
plt.title('皮马印第安人糖尿病数据LR分类的ROC和AUC', fontsize=20)
plt.xlabel('FPR', fontsize=14)
plt.ylabel('TPR', fontsize=14)
plt.show()
