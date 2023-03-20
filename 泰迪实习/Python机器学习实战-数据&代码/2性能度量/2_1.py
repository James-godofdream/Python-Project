from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_true = [1, 0, 1, 1, 0]   # 样本实际值
y_pred = [1, 0, 1, 0, 0]   # 模型预测值
res = precision_score(y_true, y_pred, average=None)   # 准确率
res = confusion_matrix(y_true, y_pred)
res = classification_report(y_true, y_pred)

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris   # 导入鸢尾花数据集
from sklearn.svm import SVC
iris = load_iris()
clf = SVC(kernel='linear', C=1)   # 构建支持向量机模型
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(scores)



print(res)



