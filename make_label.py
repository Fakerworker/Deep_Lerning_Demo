# 神经网络的搭建--分类任务 #
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F   # 激励函数都在这
import numpy as np
# x0，x1是数据，y0,y1是标签
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = np.random.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = np.zeros(100,dtype=np.int)               # 类型0 y data (tensor), shape=(100, )
x1 = np.random.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = np.ones(100,dtype=np.int)                # 类型1 y data (tensor), shape=(100,

print(y1)

Y = np.append(y0,y1)
print(Y)
for i in range(200):
    print(Y[i])