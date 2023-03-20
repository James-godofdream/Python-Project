import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):    # 网络激活函数
    return 1/(1+np.exp(-x))


data_tr = pd.read_csv('BPdata_tr.txt')  # 训练集样本
data_te = pd.read_csv('BPdata_te.txt')  # 测试集样本
n = len(data_tr)
yita = 0.85  # 学习速率

out_in = np.array([0.0, 0, 0, 0, -1])   # 输出层的输入
w_mid = np.zeros([3, 4])  # 隐层神经元的权值&阈值
w_out = np.zeros([5])     # 输出层神经元的权值&阈值

delta_w_out = np.zeros([5])      # 输出层权值&阈值的修正量
delta_w_mid = np.zeros([3, 4])   # 中间层权值&阈值的修正量
Err = []
'''
模型训练
'''
for j in range(1000):
    error = []
    for it in range(n):
        net_in = np.array([data_tr.iloc[it, 0], data_tr.iloc[it, 1], -1])  # 网络输入
        real = data_tr.iloc[it, 2]
        for i in range(4):
            out_in[i] = sigmoid(sum(net_in * w_mid[:, i]))  # 从输入到隐层的传输过程
        res = sigmoid(sum(out_in * w_out))   # 模型预测值
        error.append(abs(real-res))

        # print(it, '个样本的模型输出：', res, 'real:', real)
        delta_w_out = yita*res*(1-res)*(real-res)*out_in  # 输出层权值的修正量
        delta_w_out[4] = -yita*res*(1-res)*(real-res)     # 输出层阈值的修正量
        w_out = w_out + delta_w_out   # 更新

        for i in range(4):
            delta_w_mid[:, i] = yita*out_in[i]*(1-out_in[i])*w_out[i]*res*(1-res)*(real-res)*net_in   # 中间层神经元的权值修正量
            delta_w_mid[2, i] = -yita*out_in[i]*(1-out_in[i])*w_out[i]*res*(1-res)*(real-res)         # 中间层神经元的阈值修正量
        w_mid = w_mid + delta_w_mid   # 更新
    Err.append(np.mean(error))
plt.plot(Err)
plt.show()
plt.close()

'''
将测试集样本放入训练好的网络中去
'''
error_te = []
for it in range(len(data_te)):
    net_in = np.array([data_te.iloc[it, 0], data_te.iloc[it, 1], -1])  # 网络输入
    real = data_te.iloc[it, 2]
    for i in range(4):
        out_in[i] = sigmoid(sum(net_in * w_mid[:, i]))  # 从输入到隐层的传输过程
    res = sigmoid(sum(out_in * w_out))   # 模型预测值
    error_te.append(abs(real-res))
plt.plot(error_te)
plt.show()
np.mean(error_te)


from sklearn.neural_network import MLPRegressor

'''
调用sklearn实现神经网络算法
'''

data_tr = pd.read_csv('BPdata_tr.txt')  # 训练集样本
data_te = pd.read_csv('BPdata_te.txt')  # 测试集样本

model = MLPRegressor(hidden_layer_sizes=(10,), random_state=10, max_iter=800, learning_rate_init=0.3)  # 构建模型
model.fit(data_tr.iloc[:, :2], data_tr.iloc[:, 2])    # 模型训练
pre = model.predict(data_te.iloc[:, :2])              # 模型预测
err = np.abs(pre - data_te.iloc[:, 2]).mean()         # 模型预测误差
err










