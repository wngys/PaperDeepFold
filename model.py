import torch
import torch.nn as nn
kernel_List = [12, 4, 4, 4, 4, 4]
channel_List = [128, 256, 512, 512, 512, 400]

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_sz, padding, stride = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_sz, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.dropout(x)
        return x

def get_convBlocks(in_channel):
    layerNum = len(kernel_List)
    blocks = []
    blocks.append(ConvBlock(in_channel, channel_List[0], kernel_List[0], int(kernel_List[0] / 2 - 1)))
    for i in range(1, layerNum):
        blocks.append(ConvBlock(channel_List[i-1], channel_List[i], kernel_List[i], int(kernel_List[i] / 2 - 1)))
    return blocks

class DeepFold(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.convLayer = nn.Sequential(*get_convBlocks(in_channel))
    
    # [batch_size, 3, 256, 256]
    def forward(self, x):
        # [batch_size, 400, 4, 4]
        x = self.convLayer(x)
        # [batch_size, 400, 4]
        x = torch.diagonal(x, dim1=2, dim2=3)
        # [batch_size, 400]
        x = torch.mean(x, dim= 2)

        normValue = torch.norm(x, dim = 1) # norm_value [batch_size]
        # print(normValue.shape)
        # [400, batch_size]  最后一维要和norm_value维度匹配
        x = x.reshape(x.shape[-1], -1)
        # [400, batch_size] 已经正则化
        x = torch.div(x, normValue)

        # [batch_size, 400]
        x = x.view(x.shape[-1], -1)
        return x