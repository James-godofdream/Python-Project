from data_process import get_img_data     # 导入数据预处里的函数
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


data, labels = get_img_data()    # 数据预处理

data_tr, data_te, labels_tr, labels_te = train_test_split(data, labels, test_size=0.2)  # 将专家样本拆分为训练集和测试集
Dtc = DecisionTreeClassifier().fit(data_tr, labels_tr)  # 模型训练
pre = Dtc.predict(data_te)   # 模型预测

sum(pre==labels_te)/len(pre)            # 预测精度
confusion_matrix(labels_te, pre)        # 混淆矩阵
classification_report(labels_te, pre)   # 分类性能报告


