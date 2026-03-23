import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as scio0

from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
import csv
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import scipy
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
device = torch.device("cuda")
# 假设你的CSV文件名为 'data.csv'，并且位于与此Python脚本相同的目录下
file_path = 'ETTh1.csv'
# file_path1 = 'electricity.csv'

# 读取CSV文件
df = pd.read_csv(file_path)
flow_data = df.to_numpy()[0:,1:].T
flow_data = np.array(flow_data,dtype=np.float32)
# 显示DataFrame的前几行，以检查数据是否正确读入
print("flow_data的形状为：", flow_data.shape)
norm = np.linalg.norm(flow_data)
# 对向量进行归一化处理
flow_data = (flow_data / norm).T
flow_data.shape
print("flow_data的形状为：", flow_data.shape)
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio0
from torch.utils.data import Dataset
class LoadData(Dataset):
    def __init__(self, num_nodes, divide_days, time_interval, history_length, train_mode):


        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_days[0]  # 420天
        self.val_days = divide_days[1]  # 30天
        self.test_days = divide_days[2]
        self.history_length = history_length  # 10
        self.time_interval = time_interval  # 60min
        self.one_day_length = int(24 * 60 / self.time_interval)
#         self.graph = A
        #self.graph = A
        #对数据进行预处理，做一个归一化，norm_dim定义了在哪一个维度上进行归一化
        self.flow_data = flow_data

    def __len__(self):
   
        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length
        elif self.train_mode == "val":
            return self.val_days * self.one_day_length - self.history_length
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length-720
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):  # (x, y), index = [0, L1 - 1]

        if self.train_mode == "train":
            index = index
        elif self.train_mode == "val":
            index += self.train_days * self.one_day_length
        elif self.train_mode == "test":
            index += (self.train_days+self.val_days) * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData.slice_data(self.flow_data, self.history_length, index, self.train_mode)

        data_x = LoadData.to_tensor(data_x).to(device)# [N, H, D]
        data_y = LoadData.to_tensor(data_y).to(device)# [N, 1, D]

        return { "flow_x": data_x, "flow_y": data_y}
   
    """
    定义一些辅助函数
    
    """
        
    def slice_data(data, history_length, index, train_mode):

        if train_mode == "train":
            start_index = index
            end_index = index + history_length
        elif train_mode == "val":
            start_index = index
            end_index = index + history_length
        elif train_mode == "test":
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError("train model {} is not defined".format(train_mode))

        data_x = data[start_index: end_index,:] #[N,5,D]
        data_y = data[end_index:end_index+720,:]

        return data_x, data_y
     
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
from torch.utils.data import DataLoader
# 加载数据
train_data = LoadData(num_nodes=7, divide_days=[10452,3484,3484],
                          time_interval=60*24, history_length=96,
                          train_mode="train")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)

val_data = LoadData(num_nodes=7, divide_days=[10452,3484,3484],
                          time_interval=60*24, history_length=96,
                          train_mode="val")

val_loader = DataLoader(val_data, batch_size=5273, shuffle=False, num_workers=0)

test_data = LoadData( num_nodes=7, divide_days=[10452,3484,3484],
                          time_interval=60*24, history_length=96,
                          train_mode="test")

test_loader = DataLoader(test_data, batch_size=691, shuffle=False, num_workers=0)


print(len(train_data))
print(train_data[1190]["flow_x"].size())
print(train_data[1190]["flow_y"].size())
print(len(val_data))
print(val_data[390]["flow_x"].size())
print(val_data[390]["flow_y"].size())
print(len(test_data))
print(test_data[13216]["flow_x"].size())
print(test_data[13216]["flow_y"].size())
