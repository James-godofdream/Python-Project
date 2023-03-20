from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

iris = load_iris()

data_tr, data_te, label_tr, label_te = train_test_split(iris.data, iris.target, test_size=0.2)
model = LinearSVC().fit(data_tr, label_tr)
pre = model.predict(data_te)
acc_te = sum(pre == label_te)/len(pre)

