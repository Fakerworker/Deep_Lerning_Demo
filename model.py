import torch
import torch.nn.functional as F
import torch.nn as nn
class BP_2Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_output):
        super(BP_2Net, self).__init__()     # 继承 __init__ 功能
        self.fc1 = torch.nn.Linear(n_feature, 10)   # 隐藏层线性输出
        self.fc2 = torch.nn.Linear(10, 20)
        self.fc3 = torch.nn.Linear(20, 10)
        self.fc4 = torch.nn.Linear(10, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.fc1(x))      # 激励函数(隐藏层的线性值)
        x = F.dropout(x, p=0.3)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.3)
        x = self.fc4(x)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()