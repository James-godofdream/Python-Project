import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv('LogisticRegression.csv')
data_tr, data_te, label_tr, label_te = train_test_split(data.iloc[:, 1:], data['admit'], test_size=0.2)
clf = LogisticRegression()
clf.fit(data_tr, label_tr)
pre = clf.predict(data_te)
res = classification_report(label_te, pre)
print(res)
