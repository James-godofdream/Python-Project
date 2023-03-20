from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])  # 模型训练
'''
y = 0.5*x1 + 0.5*x2
'''
pre = clf.predict([[3, 3]])   # 模型预测
clf.coef_
clf.intercept_
print(pre)

# 波士顿房价数据回归分析
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
bosten = load_boston()  # 实例化
x = bosten.data[:, 5:6]

clf = LinearRegression()
clf.fit(x, bosten.target)   # 模型训练
clf.coef_   # 回归系数
y_pre = clf.predict(x)   # 模型输出值

plt.scatter(x, bosten.target)  # 样本实际分布
plt.plot(x, y_pre, color='red')   # 绘制拟合曲线
plt.show()












