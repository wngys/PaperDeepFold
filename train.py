# 训练过程
#/home/wngys/lab/DeepFold/Code
import random
from model import *
from data import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 1, 2, 3"
device_ids = [0, 1, 2, 3]

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
#---------------------------------------------------------------------------

class Max_margin_loss(nn.Module):
    def __init__(self, K, m) -> None:
        super().__init__()
        self.K = K
        self.m = m
   
    def forward(self, fingerpvec1, fingerpvec2, label):
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
            posi_cos_smi_list.append(F.cosine_similarity(fingerpvec1, posi_vec, dim = 0))
        for nega_vec in nega_vec_list:
            nega_cos_smi_list.append(F.cosine_similarity(fingerpvec1, nega_vec, dim = 0))

        posi_cos_smi_list.sort() # 升序排序 选最小
        nega_cos_smi_list.sort(reverse=True) # 降序排序 选最大
        # posi_cos = posi_cos_smi_list[0] # 只选取一个正例
        # loss = 0
        # for i in range(self.K):
        #     nega_cos = nega_cos_smi_list[i]
        #     loss += max(0, nega_cos - posi_cos + self.m)


        # loss = 0
        # for posi_cos in posi_cos_smi_list[:1]:
        #     for nega_cos in nega_cos_smi_list[:self.K]:
        #         # if nega_cos - posi_cos + self.m < 0:
        #         #     print("loss_toadd: ", nega_cos - posi_cos + self.m)
        #         # if nega_cos - posi_cos + self.m >= 0:
        #         loss += max(0, nega_cos - posi_cos + self.m)

        loss = 0
        for posi_cos in posi_cos_smi_list:
            for nega_cos in nega_cos_smi_list:
                loss += max(0, nega_cos - posi_cos + self.m)
        print(loss)
        return loss

def valid_data(best_acc, valid_acc_ls, dict_data, validIDlist, DFold_model, train_tfm, batch_size, device, epoch, K=5):

    valid_pair_dir = "/home/wngys/lab/DeepFold/pair/pair_bool_90/" 
    DFold_model.eval()
    acc_num = 0
    for id in validIDlist:
        feature1 = get_feature(dict_data, id, train_tfm)
        feature1 = feature1.to(device)

        id_list = get_id_list(valid_pair_dir + id +".txt")
        train_ds = Train_set(dict_data, id_list, train_tfm)
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)

        with torch.no_grad():
            fingerpvec1 = DFold_model(feature1)

        topKList = []
        for feature2, label in train_dl:
            feature2 = feature2.to(device)
            label = label.to(device)
            with torch.no_grad():
                fingerpvec2 = DFold_model(feature2)
            cos_smi_list = []
            for i in range(fingerpvec2.shape[0]):
                cos_smi_list.append(F.cosine_similarity(fingerpvec1, fingerpvec2[i], dim=-1))

            for cos_smi_idx, cos_smi in enumerate(cos_smi_list):
                if(len(topKList) < K):
                    topKList.append((cos_smi, label[cos_smi_idx]))
                else:
                    min_value = min(topKList)
                    if (cos_smi, label[cos_smi_idx]) > min_value:
                        min_idx = topKList.index(min_value)
                        topKList[min_idx] = (cos_smi, label[cos_smi_idx])
        acc_flag = False
        for _, label in topKList:
            if label == 1:
                acc_flag = True
        if acc_flag:
            acc_num += 1

    acc = acc_num / len(validIDlist)
    valid_acc_ls.append(acc)
    print(f"Epoch: {epoch} | acc_num: {acc_num} | total_num: {len(validIDlist)} | acc: {acc_num / len(validIDlist):.4f}")
    if best_acc >= 0: # 验证集
        if acc > best_acc:
            best_acc = acc
            torch.save(DFold_model.state_dict(), "/home/wngys/lab/DeepFold/model/model_oneC/best_model.pt")
            print(f"saving best model with acc: {best_acc:.4f}")    
    DFold_model.train()
    return best_acc

# 训练过程
#/home/wngys/lab/DeepFold/Code

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DFold_model = DeepFold(in_channel = 1)
# DFold_model.to(device)
DFold_model = nn.DataParallel(DFold_model, device_ids).to(device)

train_tfm = build_transform()
optimizer = torch.optim.SGD(DFold_model.parameters(), lr = 1e-3, momentum=0.9)
lossF = Max_margin_loss(K = 10, m = 2)

total_epochs = 20
batch_size = 64

train_loss_ls = []
valid_acc_ls = []
train_acc_ls = []

pair_dir = "/home/wngys/lab/DeepFold/pair/train_pair_bool_90/"  
# valid_pair_dir = "/home/wngys/lab/DeepFold/pair/pair_bool_90/"

trainIDlist = np.load("/home/wngys/lab/DeepFold/pair/train.npy", allow_pickle=True)
# random.shuffle(trainIDlist)
trainIDlist = trainIDlist[:400]
validIDlist = np.load("/home/wngys/lab/DeepFold/pair/valid.npy", allow_pickle=True)
random.shuffle(validIDlist)
validIDlist = validIDlist[:100]

dict_data = np.load("/home/wngys/lab/DeepFold/distance_matrix_r/matrix_data_1.npy", allow_pickle=True).tolist()
# resume_dir = None
resume_dir = "/home/wngys/lab/DeepFold/model/model_oneC/model_12.pt"
if resume_dir is not None:
    chkp = torch.load(resume_dir)
    st_epoch = chkp["epoch"]
    best_acc = chkp["best_acc"]
    train_loss_ls.extend(chkp["train_loss_ls"])
    valid_acc_ls.extend(chkp["valid_acc_ls"])
    train_acc_ls.extend(chkp["train_acc_ls"])
    DFold_model.load_state_dict(chkp["model_param"])
    optimizer.load_state_dict(chkp["optim_param"])
    validIDlist = chkp["valid_id_list"]
else:
    st_epoch = 0
    best_acc = 0


for epoch in range(st_epoch, total_epochs):
    # 遍历左侧一列集合每一个Protein ID
    DFold_model.train()
    total_train_loss = 0

    # random.shuffle(trainIDlist)
    for id_idx, id in enumerate(trainIDlist):
        feature1 = get_feature(dict_data, id, train_tfm)
        feature1 = feature1.to(device)

        id_list = get_id_list(pair_dir + id +".txt")
        train_ds = Train_set(dict_data, id_list, train_tfm)
        train_dl = DataLoader(train_ds, batch_size, shuffle=False, num_workers=2, pin_memory=True)

        IDtotalLoss = 0
        for feature2, label in train_dl:
            fingerpvec1 = DFold_model(feature1)
            fingerpvec1 = fingerpvec1.squeeze(0)
            feature2 = feature2.to(device)
            label = label.to(device)
            fingerpvec2 = DFold_model(feature2)
            
            loss = lossF(fingerpvec1, fingerpvec2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            IDtotalLoss += loss.item()
        
        total_train_loss += IDtotalLoss

        print(f"Epoch: {epoch} | IDidx: {id_idx} | queryID: {id} | avg_loss: {IDtotalLoss / len(train_ds):.4f} | pair_num: {len(train_ds)}")

        if (id_idx + 1) % 200 == 0:
            print("valid_Set:")
            best_acc = valid_data(best_acc, valid_acc_ls, dict_data, validIDlist, DFold_model, train_tfm, batch_size, device, epoch, K=5)
            print("train_Set:")
            valid_data(-1, train_acc_ls, dict_data, trainIDlist, DFold_model, train_tfm, batch_size, device, epoch, K=5)

    train_loss_ls.append(total_train_loss)
    print(f"Epoch: {epoch} | total_loss: {total_train_loss:.4f}")

    chkp = {
        "epoch": epoch+1,
        "best_acc": best_acc,
        "model_param": DFold_model.state_dict(),
        "optim_param": optimizer.state_dict(),
        "train_loss_ls": train_loss_ls,
        "valid_acc_ls": valid_acc_ls,
        "train_acc_ls": train_acc_ls,
        "valid_id_list": validIDlist
    }
    torch.save(chkp, "/home/wngys/lab/DeepFold/model/model_oneC/" + f"model_{epoch}.pt")
        

            
                    
                    




        



