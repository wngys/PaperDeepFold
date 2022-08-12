#/home/wngys/lab/DeepFold/Code
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import *

import os

# --------------------------------------------------------------------------------------------- #
# 设置显卡
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
device_ids = [0, 1, 2, 3]

# model_path = "/home/wngys/lab/DeepFold/new_model/new_model/model_2.pt"
model_path = "/home/wngys/lab/DeepFold/new_model/new_model_3/model_9.pt"
chkp = torch.load(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DFold_model = DeepFold(in_channel = 1)
DFold_model = DeepFold(in_channel = 3)
DFold_model = nn.DataParallel(DFold_model, device_ids).to(device)
DFold_model.load_state_dict(chkp["model_param"])

optimizer = torch.optim.SGD(DFold_model.parameters(), lr = 1e-3, momentum=0.9)

# --------------------------------------------------------------------------------------------- #
# 自定义Dataset：加载蛋白质id、距离矩阵、标签
class MatrixLabelDataset(Dataset):
    def __init__(self, protein_id, pair_dir, matrix_dir, transform=None):
        self.protein_id = protein_id
        self.matrix_dir = matrix_dir
        self.transform = transform
        pair_path = pair_dir + protein_id + ".txt"
        self.id_label_list = self.get_id_label_list(pair_path)

    def __len__(self):
        return len(self.id_label_list)

    def __getitem__(self, idx):
        id = self.id_label_list[idx][0]
        label = self.id_label_list[idx][1]
        matrix_path = self.matrix_dir + id + ".npy"
        # matrix = torch.from_numpy(np.expand_dims(np.load(matrix_path, allow_pickle=True), 0)).to(torch.float)
        matrix = torch.from_numpy(np.load(matrix_path, allow_pickle=True)).to(torch.float)
        if self.transform:
            matrix = self.transform(matrix)
        return id, matrix, label
    
    def get_id_label_list(self, pair_path):
        id_label_list = []
        with open(pair_path, "r") as f_r:
            while True:
                lines = f_r.readline()
                if not lines:
                    break
                id= lines.split('\t')[0]
                label = lines.split('\t')[1].split("\n")[0]
                id_label_list.append((id, label))
        return id_label_list

# --------------------------------------------------------------------------------------------- #
# 计算loss
def MaxMarginLoss(vectors):
    query_vector = vectors[:1]
    pos_vectors = vectors[1:7]
    neg_vectors = vectors[7:]

    query_vector_6 = query_vector.repeat(6, 1)
    query_vector_57 = query_vector.repeat(57, 1)

    pos_cos_simi = F.cosine_similarity(query_vector_6, pos_vectors, dim=1).view(6, 1)
    neg_cos_simi = F.cosine_similarity(query_vector_57, neg_vectors, dim=1).view(1, 57)

    m = 0.1
    diff = neg_cos_simi - pos_cos_simi + m
    loss = torch.sum(diff[diff>=0])
    # print(loss)
    
    return loss

# --------------------------------------------------------------------------------------------- #
def by_simi(t):
    return t[2]

# --------------------------------------------------------------------------------------------- #
# 模型在训练过程中，测试在验证集上的准确率acc
def ModelOnValidSet(valid_pair_dir, valid_matrix_dir, K):
    DFold_model.eval()  
    cntShot = 0
    for idx, protein_id in enumerate(validIDlist):
        # print(protein_id)
        query_matrix_path = valid_matrix_dir + protein_id + ".npy"
        query_matrix = torch.unsqueeze(torch.from_numpy(np.load(query_matrix_path, allow_pickle=True)).to(torch.float), 0)
        # query_matrix = torch.unsqueeze(torch.from_numpy(np.expand_dims(np.load(query_matrix_path, allow_pickle=True), 0)).to(torch.float), 0)
        query_matrix = transform(query_matrix)
        query_matrix = query_matrix.to(device)
        query_vector = DFold_model(query_matrix)
        
        valid_dataset = MatrixLabelDataset(protein_id, valid_pair_dir, valid_matrix_dir, transform)
        valid_dataloader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        
        id_label_simi_list = []

        for ids, matrices, labels in valid_dataloader:
            bs = len(ids) # batch size
            matrices = matrices.to(device)
            vectors = DFold_model(matrices)
            query_vector_bs = query_vector.repeat(bs, 1)
            cos_simi = F.cosine_similarity(query_vector_bs, vectors, dim=1)

            for i in range(bs):
                id_label_simi_list.append((ids[i], labels[i], cos_simi[i].tolist()))

        id_label_simi_list = sorted(id_label_simi_list, key=by_simi, reverse=True)

        shot = False
        for t in id_label_simi_list[:K]:
            if t[1] == '1':
                shot = True
                break
        if shot:
            cntShot += 1

    acc = cntShot / len(validIDlist)
    print("Valid acc:", acc, "| shot:", cntShot, "| total:", len(validIDlist))
    return (acc, cntShot, len(validIDlist))

# --------------------------------------------------------------------------------------------- #
# 模型在训练过程中，测试在训练集上的准确率acc
def ModelOnTrainSet(train_pair_dir, train_matrix_dir, K):
    DFold_model.eval()
    cntShot = 0
    for idx, protein_id in enumerate(trainIDlist[:100]):
        # print(protein_id)
        query_matrix_path = train_matrix_dir + protein_id + ".npy"
        query_matrix = torch.unsqueeze(torch.from_numpy(np.load(query_matrix_path, allow_pickle=True)).to(torch.float), 0)
        # query_matrix = torch.unsqueeze(torch.from_numpy(np.expand_dims(np.load(query_matrix_path, allow_pickle=True), 0)).to(torch.float), 0)
        query_matrix = transform(query_matrix)
        query_matrix = query_matrix.to(device)
        query_vector = DFold_model(query_matrix)
        
        train_dataset = MatrixLabelDataset(protein_id, train_pair_dir, train_matrix_dir, transform)
        train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        
        id_label_simi_list = []

        for ids, matrices, labels in train_dataloader:
            bs = len(ids) # batch size
            matrices = matrices.to(device)
            vectors = DFold_model(matrices)
            query_vector_bs = query_vector.repeat(bs, 1)
            cos_simi = F.cosine_similarity(query_vector_bs, vectors, dim=1)

            for i in range(bs):
                id_label_simi_list.append((ids[i], labels[i], cos_simi[i].tolist()))

        id_label_simi_list = sorted(id_label_simi_list, key=by_simi, reverse=True)

        shot = False
        for t in id_label_simi_list[:K]:
            if t[1] == '1':
                shot = True
                break
        if shot:
            cntShot += 1

    acc = cntShot / len(trainIDlist[:100])
    print("Train acc:", acc, "| shot:", cntShot, "| total:", len(trainIDlist[:100]))
    return (acc, cntShot, len(trainIDlist[:100]))

# --------------------------------------------------------------------------------------------- #
START_EPOCH = 10
EPOCH = 20
BATCH_SIZE = 64
K = 10

trainIDlist = np.load("/home/wngys/lab/DeepFold/pair/train.npy", allow_pickle=True)
random.shuffle(trainIDlist)
trainIDlist = trainIDlist[:1000]
# validIDlist = np.load("/home/wngys/lab/DeepFold/pair/valid.npy", allow_pickle=True)
# random.shuffle(validIDlist)
# validIDlist = validIDlist[:100]
validIDlist = chkp['valid_id_list']

train_pair_dir = "/home/wngys/lab/DeepFold/pair/new_train_pair_bool_90/"
train_matrix_dir = "/home/wngys/lab/DeepFold/distance_matrix_r/distance_matrix_mine_r_3/"
valid_pair_dir = "/home/wngys/lab/DeepFold/pair/pair_bool_90/"
valid_matrix_dir = "/home/wngys/lab/DeepFold/distance_matrix_r/distance_matrix_mine_r_3/"
transform = T.Compose([
    T.Resize((256, 256)),
    # T.Normalize(mean=[0.0660], std=[0.0467])
    T.Normalize(mean=[0.0068, 0.0003, 2.3069e-05], std=[0.0140, 0.0015, 0.0002])
])

train_acc_list = chkp['train_acc']
valid_acc_list = chkp['valid_acc']

for epoch in range(START_EPOCH, EPOCH):
    for idx, protein_id in enumerate(trainIDlist):
        # print(protein_id)
        DFold_model.train()
        train_dataset = MatrixLabelDataset(protein_id, train_pair_dir, train_matrix_dir, transform)
        train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        for ids, matrices, labels in train_dataloader:
            matrices = matrices.to(device)
            vectors = DFold_model(matrices)
            loss = MaxMarginLoss(vectors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss)
        
        if idx % 1 == 0:
            print("Epoch:", epoch, "| idx:", idx, "| id:", protein_id, "| last batch loss:", loss.tolist())
        
        if idx % 200 == 0:
            train_t = ModelOnTrainSet(train_pair_dir, train_matrix_dir, K)
            valid_t = ModelOnValidSet(valid_pair_dir, valid_matrix_dir, K)
            train_acc_list.append(train_t)
            valid_acc_list.append(valid_t)

    chkp = {
        "epoch": epoch,
        "model_param": DFold_model.state_dict(),
        "optim_param": optimizer.state_dict(),
        "train_acc": train_acc_list,
        "valid_acc": valid_acc_list,
        "valid_id_list": validIDlist
    }
    torch.save(chkp, "/home/wngys/lab/DeepFold/new_model/new_model_3/" + f"model_{epoch}.pt")