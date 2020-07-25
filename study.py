# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 23:03:33 2018
@author: xxx
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 由于LogiReg_data.txt文件第一行不是列名，所以不让header=None，自己定义列名names
pdData = pd.read_csv("LogiReg_data.txt", header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

"""#可以把数据画在坐标轴上看一下
positive = pdData[pdData['Admitted'] == 1]
negative = pdData[pdData['Admitted'] == 0]
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
"""


# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


'''可以看一下这些数据的sigmoid函数图像
nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(nums, sigmoid(nums), 'r')
'''


def model(X, theta):  # 向量版sigmoid函数
    return sigmoid(np.dot(X, theta.T))


pdData.insert(0, 'Ones', 1)  # 往X中第一列插入1
# print(pdData)
orig_data = pdData.values  # 除去了表名与索引
# print(orig_data)
cols = orig_data.shape[1]  # 查看orig_data的列数，返回cols=4
# print(cols)
X = orig_data[:, 0:cols - 1]
y = orig_data[:, cols - 1:cols]
theta = np.zeros([1, 3])  # 构造全零theta


# 损失函数
def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))


# print(cost(X, y, theta))
# print(len(X))
# print(X)

# 计算梯度
def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)
    return grad


# 不同梯度下降方法，停止策略
STOP_ITER = 0  # 按照次数进行停止
STOP_COST = 1  # 根据前后差异，若很小就可以停止
STOP_GRAD = 2


def stopCriterion(type, value, threshold):  # threshold是阈值
    if type == STOP_ITER:
        return value > threshold
    elif type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold  # value[-2]表示倒数第二个元素
    elif type == STOP_GRAD:
        return np.linalg.norm(value) < threshold  # 求范数


import numpy.random


# 洗牌,重新打乱数据
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols - 1]
    y = data[:, cols - 1:cols]  #####
    return X, y


import time


# 梯度下降求解
def descent(data, theta, batchSize, stopType, thresh, alpha):
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)  # 计算梯度
    costs = [cost(X, y, theta)]  # 损失值
    print('1')
    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
        k += batchSize  # 取batch数量个数据
        if k >= n:
            k = 0
            X, y = shuffleData(data)  # 重新洗牌
        theta = theta - alpha * grad  # 参数更新
        costs.append(cost(X, y, theta))  # 计算新的损失
        i += 1

        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopCriterion(stopType, value, thresh): break
    print("10")
    return theta, i - 1, costs, grad, time.time() - init_time


# 计算并画图
def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)  # 初始化
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += "data - learning rate: {} -".format(alpha)
    if batchSize == n:
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + "descent - Stop: "
    if stopType == STOP_ITER:
        strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST:
        strStop = "costs change < {}".format(thresh)
    else:
        strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print("***{}\nTheta:{} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(name, theta, iter, costs[-1],
                                                                                          dur))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel("Iterations")
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iterations')
    plt.show()
    return theta


n = 100  # 基于所有样本，即全部100个数据
#根据迭代次数停止，这里设置了5000次
#runExpe(orig_data, theta, n, STOP_ITER, thresh=50000, alpha=0.000001)
runExpe(orig_data, theta, n, STOP_ITER, thresh=250000, alpha=0.001)
# 根据损失值停止
#runExpe(orig_data, theta, n, STOP_COST, thresh=0.0000001, alpha=0.001)


# 根据梯度变化停止
# runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.05, alpha=0.001)

# 设定精度
def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]


predictions = predict(X, theta)
correct = [1 if ((a == 1 and b == 1) or a == 0 and b == 0) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))

# 如果再加上数据预处理，精度会好很多，90%左右
