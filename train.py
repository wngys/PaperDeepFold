# 训练过程
#/home/wngys/lab/DeepFold/Code
from model import *
from data import *
from torch.utils.data import DataLoader

# 获取左侧一列id 对应的右侧id_list
def get_id_list(pair_path):
    id_list = []
    with open(pair_path, "r") as f_r:
        while True:
            lines = f_r.readline()
            if not lines:
                break
            line = lines.split('\n')[0].split('\t')
            id_list.append((line[0], line[1]))

    return id_list

# 获取id对应的distance_matrix
def get_feature(data_path):
    feature = torch.from_numpy(np.load(data_path, allow_pickle=True))
    feature = torch.unsqueeze(feature, 0)
    return feature


#---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DFold_model = DeepFold(in_channel = 3)
DFold_model.to(device)

train_tfm = build_transform(in_channel=3)

# print(train_tfm)
optimizer = torch.optim.SGD(DFold_model.parameters(), lr = 1e-3)

total_epochs = 100
batch_size = 64

resume_dir = None
if resume_dir is not None:
    pass
else:
    st_epoch = 0

pair_dir = "../pair/pair_bool/"  
data_dir = "../distance_matrix/distance_matrix_inf/" 

for epoch in range(st_epoch, total_epochs):
    # 遍历左侧一列集合每一个Protein ID
    for id in trainIDset:
        id_list = get_id_list(pair_dir + id +".txt")
        feature1 = get_feature(data_dir + id + ".npy")
        feature1 = feature1.to(device)

        train_ds = Train_set(data_dir, id_list, train_tfm)
        train_dl = DataLoader(train_ds, batch_size, shuffle=False, num_workers=2, pin_memory=True)

        fingerpvec1 = DFold_model(feature1)

        for feature2, label in train_dl:
            feature2 = feature2.to(device)
            label = label.to(device)
            fingerpvec2 = DFold_model(feature2)
            
            posi_vec_list = []
            nega_vec_list = []

            for number_inbatch in range(fingerpvec2.shape[0]):
                if label[number_inbatch] == 0:
                    nega_vec_list.append(fingerpvec2[number_inbatch])
                elif label[number_inbatch] == 1:
                    posi_vec_list.append(fingerpvec2[number_inbatch])
                else:
                    print("ERROR")