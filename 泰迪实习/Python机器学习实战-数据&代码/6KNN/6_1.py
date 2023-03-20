from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()  # 鸢尾花数据
data_tr, data_te, label_tr, label_te = train_test_split(iris.data, iris.target, test_size=0.2)   # 拆分专家样本集

model = KNeighborsClassifier(n_neighbors=5)   # 构建模型
model.fit(data_tr, label_tr)   # 模型训练
pre = model.predict(data_te)   # 模型预测
acc = model.score(data_te, label_te)   # 模型在测试集上的精度
acc



