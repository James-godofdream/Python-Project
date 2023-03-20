import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report
import graphviz

data = pd.read_csv('titanic_data.csv')
data.drop('PassengerId', axis=1, inplace=True)   # 删除PassengerId 列

data.loc[data['Sex'] == 'male', 'Sex'] = 1       # 用数值1来代替male，用0来代替female
data.loc[data['Sex'] == 'female', 'Sex'] = 0
data.fillna(data['Age'].mean(), inplace=True)    # 用均值来填充缺失值

Dtc = DecisionTreeClassifier(max_depth=5, random_state=8)    # 构建决策树模型
Dtc.fit(data.iloc[:, 1:], data['Survived'])       # 模型训练
pre = Dtc.predict(data.iloc[:, 1:])               # 模型预测
pre == data['Survived']                           # 比较模型预测值与样本实际值是否一致
classification_report(data['Survived'], pre)      # 分类报告

dot_data = export_graphviz(Dtc, feature_names=['Pclass', 'Sex', 'Age'], class_names='Survived')
graph = graphviz.Source(dot_data)       # 决策树可视化
graph

