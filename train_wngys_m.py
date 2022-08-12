import random
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn as nn
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
device_ids = [0, 1, 2, 3]
#----------------------------------------------------------------------------------------
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
    def forward(self, x):
        x = self.convLayer(x)
        x = torch.diagonal(x, dim1=2, dim2=3) # [batch_size, 400, 4]
        x = torch.mean(x, dim= 2)  # [batch_size, 400]
        x = F.normalize(x) 
        return x
#----------------------------------------------------------------------------------------
def build_transform():
    train_tfm = T.Compose(
        [
            T.Resize((256, 256)),
            # T.Normalize(mean=[0.0068, 0.0003, 2.3069e-05], std=[0.0140, 0.0015, 0.0002])
            T.Normalize(mean=[0.0660], std=[0.0467])
        ]
    )
    return train_tfm

class Train_set(torch.utils.data.Dataset):
    def __init__(self, dict_data, id_list, tfm) -> None:
        super().__init__()
        self.tensor_list = []
        for id, label in id_list:
            feature = torch.from_numpy(dict_data[id])
            self.tensor_list.append((feature, label))
        self.tfm = tfm

    def __getitem__(self, idx :int):
        x = self.tensor_list[idx][0]
        x = x.to(torch.float)
        x = self.tfm(x)
        label = self.tensor_list[idx][1]
        return x,label

    def __len__(self):
        return len(self.tensor_list)
#----------------------------------------------------------------------------------------
class Max_margin_loss(nn.Module):
    def __init__(self, K, m) -> None:
        super().__init__()
        self.K = K
        self.m = m

    def forward(self, fpvec1, fpvec2):
        # vec1 [1,400]
        # vec2 [64,400]
        pos_vec = fpvec2[:6]
        neg_vec = fpvec2[6:]
        fpvec1_6 = fpvec1.repeat(6, 1)
        fpvec1_58 = fpvec1.repeat(58,1)
        pos_cos = F.cosine_similarity(fpvec1_6, pos_vec, dim=-1).view(6, 1)
        neg_cos = F.cosine_similarity(fpvec1_58, neg_vec, dim=-1).view(1, 58)

        diff = neg_cos - pos_cos + self.m
        loss = torch.sum(diff[diff>=0])
        return loss
#----------------------------------------------------------------------------------------
# 获取左侧一列id 对应的右侧id_list
def get_id_list(pair_path):
    id_list = []
    with open(pair_path, "r") as f_r:
        while True:
            lines = f_r.readline()
            if not lines:
                break
            line1= lines.split('\t')[0]
            line2 = lines.split('\t')[1].split("\n")[0]
            id_list.append((line1, line2))
    return id_list

# # 获取id对应的distance_matrix
def get_feature(dict_data, id, tfm):
    feature = torch.from_numpy(dict_data[id])
    feature = feature.to(torch.float)
    feature = tfm(feature)
    feature = feature.unsqueeze(0)
    return feature
#----------------------------------------------------------------------------------------
chkp = torch.load("/home/wngys/lab/DeepFold/model/model_lossF/model_9.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DFold_model = DeepFold(in_channel = 1)
DFold_model = nn.DataParallel(DFold_model, device_ids).to(device)
DFold_model.load_state_dict(chkp["model_param"])
train_tfm = build_transform()
optimizer = torch.optim.SGD(DFold_model.parameters(), lr = 1e-3, momentum=0.9)
lossF = Max_margin_loss(K = 10, m = 0.1)

st_epoch = 10
total_epochs = 20
batch_size = 64

train_loss_ls = chkp["train_loss"]
valid_acc_ls = chkp["valid_acc"]
train_acc_ls = chkp["train_acc"]

trainIDlist = np.load("/home/wngys/lab/DeepFold/pair/train.npy", allow_pickle=True)
random.shuffle(trainIDlist)
trainIDlist = trainIDlist[:400]
# validIDlist = np.load("/home/wngys/lab/DeepFold/pair/valid.npy", allow_pickle=True)
# random.shuffle(validIDlist)
# validIDlist = validIDlist[:100]
validIDlist = chkp["valid_id_list"]
pair_dir = "/home/wngys/lab/DeepFold/pair/train_pair_bool_90/"  
dict_data = np.load("/home/wngys/lab/DeepFold/distance_matrix_r/matrix_data_1.npy", allow_pickle=True).tolist()
#----------------------------------------------------------------------------------------
def valid_test(mode):
    valid_pair_dir = "/home/wngys/lab/DeepFold/pair/pair_bool_90/" 
    DFold_model.eval()
    K = 5
  
    acc_num = 0

    if mode == 0:
        IDlist = trainIDlist[:100]
    else:
        IDlist = validIDlist

    for id in IDlist:
        feature1 = get_feature(dict_data, id, train_tfm)
        feature1 = feature1.to(device)

        id_list = get_id_list(valid_pair_dir + id +".txt")
        train_ds = Train_set(dict_data, id_list, train_tfm)
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
        with torch.no_grad():
            fpvec1 = DFold_model(feature1)
        topList = []
        for feature2, label in train_dl:
            feature2 = feature2.to(device)
            with torch.no_grad():
                fpvec2 = DFold_model(feature2)
            fpvec1_bs = fpvec1.repeat(fpvec2.shape[0], 1)
            cos_sim = F.cosine_similarity(fpvec1_bs, fpvec2, dim=-1)
            
            for i in range(fpvec2.shape[0]):
                topList.append((cos_sim[i], label[i]))
        topList.sort(reverse=True)
        acc_flag = False
        for _, label in topList[:K]:
            if label == '1':
                acc_flag = True
                break
        if acc_flag:
            acc_num += 1

    acc = acc_num / len(IDlist)
    print(f"acc: {acc} | acc_num: {acc_num} | total: {len(IDlist)}")
    DFold_model.train()
    return acc
#----------------------------------------------------------------------------------------
best_acc = chkp["best_acc"]
for epoch in range(st_epoch, total_epochs):
    # 遍历左侧一列集合每一个Protein ID
    DFold_model.train()
    total_train_loss = 0
    for id_idx, id in enumerate(trainIDlist):
        feature1 = get_feature(dict_data, id, train_tfm)
        feature1 = feature1.to(device)
        id_list = get_id_list(pair_dir + id +".txt")
        train_ds = Train_set(dict_data, id_list, train_tfm)
        train_dl = DataLoader(train_ds, batch_size, shuffle=False, num_workers=2, pin_memory=True)
        IDtotalLoss = 0
        for feature2, _ in train_dl:
            fingerpvec1 = DFold_model(feature1)
            feature2 = feature2.to(device)
            fingerpvec2 = DFold_model(feature2)
            loss = lossF(fingerpvec1, fingerpvec2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            IDtotalLoss += loss.item()
        
        total_train_loss += IDtotalLoss
        print(f"Epoch: {epoch} | IDidx: {id_idx} | queryID: {id} | avg_loss: {IDtotalLoss / len(train_ds):.4f} | pair_num: {len(train_ds)}")

        if  id_idx % 200 == 0:
            print("-----train_Set-----")
            train_acc = valid_test(mode=0)
            print("-----valid_Set-----")
            valid_acc = valid_test(mode=1)
            train_acc_ls.append(train_acc)
            valid_acc_ls.append(valid_acc)
            if valid_acc > best_acc:
                torch.save(DFold_model.state_dict(), "/home/wngys/lab/DeepFold/model/model_lossF/best_model.pt")
                best_acc = valid_acc
                print(f"saving best_model with valid_acc: {best_acc}")

            

    train_loss_ls.append(total_train_loss)
    print(f"Epoch: {epoch} | total_loss: {total_train_loss:.4f}")

    chkp = {
        "epoch": epoch,
        "model_param": DFold_model.state_dict(),
        "optim_param": optimizer.state_dict(),
        "best_acc": best_acc,
        "train_loss": train_loss_ls,
        "train_acc": train_acc_ls,
        "valid_acc": valid_acc_ls,
        "valid_id_list": validIDlist
    }
    torch.save(chkp, "/home/wngys/lab/DeepFold/model/model_lossF/" + f"model_{epoch}.pt")

