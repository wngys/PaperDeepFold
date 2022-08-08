import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

# 数据预处理transform
#-----------------------------------------------------
#此代码块作为数据处理内容，则transform不包含此内容
# def funcy(x: torch.tensor, k:int):
#     return x**(-2*k)

# class Pretfm(torch.nn.Module):
#     def __init__(self, in_channel) -> None:
#         super().__init__()
#         self.in_channel = in_channel

#     def forward(self, x: torch.tensor):

#         y = torch.rand(self.in_channel, 256, 256)

#         for i in range(1, self.in_channel+1):
#             y[i-1] = funcy(x, i)
#         return y
#------------------------------------------------------

# 原数据应是取逆扩充channel后的距离矩阵，而且inf项被替换
def build_transform(in_channel):
    train_tfm = T.Compose(
        [
            T.Resize((256, 256)),
            # 取逆矩阵 扩充channel
            # Pretfm(in_channel),
            #是否需要数据增强 保留一个问号
            # 层归一化
            # nn.LayerNorm((in_channel, 256, 256)) # 不能经过LayNorm等网络层，不然输出数据 requires_grad = True,从而报错，原始数据应该为False
            T.Normalize(mean=[0.0068, 0.0003, 2.3069e-05], std=[0.0140, 0.0015, 0.0002])
        ]
    )
    return train_tfm
#-----------------------------------------------------

# 构建数据集
#-----------------------------------------------------
class Train_set(torch.utils.data.Dataset):

    def __init__(self, dict_data, id_list, tfm) -> None:
        super().__init__()
        self.tensor_list = []
        for id, label in id_list:
            # 在蛋白质数据库文件查找 id.npy
            # feature = torch.from_numpy(np.load(dir+id+".npy", allow_pickle=True))
            feature = torch.from_numpy(dict_data[id])
            # feature = torch.unsqueeze(feature, 0)
            label = float(label)
            self.tensor_list.append((feature,
                                        label)
                                        )
        self.tfm = tfm

    def __getitem__(self, idx :int):
        y = self.tensor_list[idx][0]
        y = y.to(torch.float)
        y = self.tfm(y)
        label = torch.tensor(self.tensor_list[idx][1], dtype=torch.float32)
        return y, label

    def __len__(self):
        return len(self.tensor_list)

#-----------------------------------------------------
class LeftTrainSet(torch.utils.data.Dataset):

    def __init__(self, dir, train_list, tfm) -> None:
        super().__init__()
        self.tensorList = []
        for leftID in train_list:
            feature = torch.from_numpy(np.load(dir+leftID+".npy", allow_pickle=True))
            # feature = torch.unsqueeze(feature, 0)
            self.tensorList.append((leftID, feature))
        self.tfm = tfm

    def __getitem__(self, idx):
        y = self.tensorList[idx][1]
        y = y.to(torch.float)
        y = self.tfm(y)
        ID = self.tensorList[idx][0]
        return ID, y

    def __len__(self):
        return len(self.tensorList)

