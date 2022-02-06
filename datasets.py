import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class Satellite_Dataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=True):
        # 读取csv文件 分别是数据和标签
        self.data_file = pd.read_csv(data_dir)
        self.label_file = pd.read_csv(label_dir)
        self.data_info = np.array(self.data_file.iloc[:,:])
        self.label_info = np.array(self.label_file.iloc[:])

        shape = self.label_info.shape[0]   # 将标签转为1维数据
        self.label_info = self.label_info.reshape(shape,)

        self.transform = transform         # 将numpy格式数据转为tensor

    def __getitem__(self, index):
        # 获得每一行数据
        data= self.data_info[index]
        label = self.label_info[index]

        # 将数据和标签转换为tensor
        if self.transform is not None:
            data = torch.from_numpy(data).type(torch.FloatTensor)
            label = torch.tensor(label)

        return data, label

    def __len__(self):
        return len(self.data_info)
