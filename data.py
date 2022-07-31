import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

# 数据预处理transform
#-----------------------------------------------------
def funcy(x: torch.tensor, k:int):
    return x**(-2*k)

class Pretfm(torch.nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.in_channel = in_channel

    def forward(self, x: torch.tensor):

        y = torch.rand(self.in_channel, 256, 256)

        for i in range(1, self.in_channel+1):
            y[i-1] = funcy(x, i)
        return y

in_channel = 3

train_tfm = T.Compose(
    [
        T.Resize((256, 256)),
        # 取逆矩阵 扩充channel
        Pretfm(in_channel),
        #是否需要数据增强 保留一个问号
        # 层归一化
        nn.LayerNorm((in_channel, 256, 256))
    ]
)
#-----------------------------------------------------

# 构建数据集
#-----------------------------------------------------
class Train_set(torch.utils.data.Dataset):
    def __init__(self, id_list, tfm) -> None:
        super().__init__()
        self.tensor_list = []
        for id in id_list:
            # 在蛋白质数据库文件查找 id.npy
            self.tensor_list.append(torch.from_numpy(np.load(dir+id+".npy", allow_pickle=True)))
        self.tfm = tfm

    def __getitem__(self, idx :int):
        y = self.tensor_list[idx]
        y = self.tfm(y)
        return y

    def __len__(self):
        return len(self.tensor_list)
