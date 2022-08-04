# 训练过程
#/home/wngys/lab/DeepFold/Code
from torch.utils import data
from model import *
from data import *
from torch.utils.data import DataLoader
import os
import csv
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./data/tensorboard')

def read_data():
    data_dir = "/home/wngys/lab/DeepFold/distance_matrix_r/distance_matrix_mine_r_3"
    data_dict = {}
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        data = np.load(file_path, allow_pickle=True)
        ID = file_name.split('.')[0]
        data_dict[ID] = data
        break

    fileName = "/home/wngys/lab/DeepFold/distance_matrix_r/matrix_data.csv"
    with open(fileName,"wb") as csv_file:
        writer=csv.writer(csv_file)
        for key,value in data_dict.items():
            writer.writerow([key,value])


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
def get_feature(data_path, tfm):
    feature = torch.from_numpy(np.load(data_path, allow_pickle=True))
    feature = feature.to(torch.float)
    feature = tfm(feature)
    feature = feature.unsqueeze(0)
    return feature

# def compute_loss(posi_cosList, nega_cosList, K = 10, m = 0.1):
#     posi_cosList.sort() # 升序排序 选最小
#     nega_cosList.sort(reverse=True) # 降序排序 选最大
#     posi_cos = posi_cosList[0] # 只选取一个正例
#     loss = 0
#     for i in range(K):
#         nega_cos = nega_cosList[i]
#         loss += max(0, nega_cos - posi_cos + m)

#     return loss
#---------------------------------------------------------------------------

class Max_margin_loss(nn.Module):
    def __init__(self, K, m) -> None:
        super().__init__()
        # self.cosFunc = F.cosine_similarity()
        self.K = K
        self.m = m
 
    
    def forward(self, fingerpvec1, fingerpvec2, label):
        # posi_idxs = []
        # nega_idxs = []
        # for i in range(len(label)):
        #     if label[i] == 1:
        #         posi_idxs.append(i)
        #     else:
        #         nega_idxs.append(i)
        posi_vec_list = []
        nega_vec_list = []

        for number_inbatch in range(fingerpvec2.shape[0]):
            if label[number_inbatch] == 0:
                nega_vec_list.append(fingerpvec2[number_inbatch])
            elif label[number_inbatch] == 1:
                posi_vec_list.append(fingerpvec2[number_inbatch])
            else:
                print("ERROR")

        posi_cos_smi_list = []
        nega_cos_smi_list = []

        for posi_vec in posi_vec_list:
            # print("posi_vec: ", posi_vec.shape)
            posi_cos_smi_list.append(F.cosine_similarity(fingerpvec1, posi_vec, dim = 0))

        for nega_vec in nega_vec_list:
            nega_cos_smi_list.append(F.cosine_similarity(fingerpvec1, nega_vec, dim = 0))

        # print(posi_cos_smi_list[0].shape)
        posi_cos_smi_list.sort() # 升序排序 选最小
        # print(nega_cos_smi_list)
        nega_cos_smi_list.sort(reverse=True) # 降序排序 选最大
        posi_cos = posi_cos_smi_list[0] # 只选取一个正例
        loss = 0
        for i in range(self.K):
            nega_cos = nega_cos_smi_list[i]
            loss += max(0, nega_cos - posi_cos + self.m)

        return loss


# 训练过程
#/home/wngys/lab/DeepFold/Code
from model import *
from data import *
from torch.utils.data import DataLoader
from torch import cosine_similarity
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DFold_model = DeepFold(in_channel = 3)
DFold_model.to(device)

train_tfm = build_transform(in_channel = 3)

# print(train_tfm)
optimizer = torch.optim.SGD(DFold_model.parameters(), lr = 1e-3)

lossF = Max_margin_loss(K = 10, m = 0.1)

total_epochs = 10
batch_size = 64

resume_dir = None
if resume_dir is not None:
    pass
else:
    st_epoch = 0

pair_dir = "/home/wngys/lab/DeepFold/pair/train_pair_bool_90/"  
data_dir = "/home/wngys/lab/DeepFold/distance_matrix_r/distance_matrix_mine_r_3/" 

theSelecTrainList = np.load("/home/wngys/lab/DeepFold/pair/train.npy", allow_pickle=True)

trainIDlist = theSelecTrainList[:64]

# leftTrain_ds = LeftTrainSet(data_dir, trainIDlist, train_tfm)
# leftTrain_dl = DataLoader(leftTrain_ds, batch_size, shuffle = True)

for epoch in range(st_epoch, total_epochs):
    # 遍历左侧一列集合每一个Protein ID
    DFold_model.train()

    for id in trainIDlist:
        feature1 = get_feature(data_dir+id+".npy", train_tfm)
        feature1 = feature1.to(device)

        id_list = get_id_list(pair_dir + id +".txt")

        train_ds = Train_set(data_dir, id_list, train_tfm)
        # print(train_ds[0][0])
        train_dl = DataLoader(train_ds, batch_size, shuffle=False, num_workers=2, pin_memory=True)

        IDtotalLoss = 0
        for feature2, label in train_dl:
            fingerpvec1 = DFold_model(feature1)
            fingerpvec1 = fingerpvec1.squeeze(0)

            feature2 = feature2.to(device)
            label = label.to(device)

            fingerpvec2 = DFold_model(feature2)
            # print(fingerpvec1.shape)
            # print(fingerpvec2.shape)
            
            loss = lossF(fingerpvec1, fingerpvec2, label)

            optimizer.zero_grad()
            
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

            IDtotalLoss += loss

        print(f"Epoch: {epoch} | queryID: {id} | avg_loss: {IDtotalLoss / len(train_dl):.4f}")

    # DFold_model.eval()