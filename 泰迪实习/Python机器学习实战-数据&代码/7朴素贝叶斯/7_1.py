import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

clf = GaussianNB()
clf.fit(X, Y)
clf.predict([[-0.8, -1]])

iris = load_iris()
data_tr, data_te, label_tr, label_te = train_test_split(iris.data, iris.target, test_size=0.2)
clf.fit(data_tr, label_tr)
pre = clf.predict(data_te)
acc = sum(pre == label_te)/len(pre)   # 模型在测试集样本上的预测精度


