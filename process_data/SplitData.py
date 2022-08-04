# 更改数据分布 确保每个batch至少一个正例
import math
import copy
import numpy as np

theSelecTrainList = np.load("/home/wngys/lab/DeepFold/pair/train.npy", allow_pickle=True)

Dir = "/home/wngys/lab/DeepFold/pair/pair_bool_90/"
newDir = "/home/wngys/lab/DeepFold/pair/train_pair_bool_90/"
batch_size = 64

# print(theSelecTrainList)
idx = 0
for id in theSelecTrainList:
    idx += 1
    if idx % 100 == 0:
        print("idx: ", idx)
    posi_data = []
    nega_data = []

    with open(Dir+id+".txt", "r") as f_r:
        while True:
            lines = f_r.readline()
            if not lines:
                break
            # id2 = lines.split('\t')[0]
            flag = lines.split('\t')[1].split("\n")[0]
            # print(type(flag))
            if flag == "1":
                posi_data.append(lines)
            else:
                nega_data.append(lines)
    
    posi_num = len(posi_data)
    nega_num = len(nega_data)
    total_num = posi_num + nega_num
    # print(f"id:{id} nega_num:{nega_num} posi_num:{posi_num}")

    # data_list = []
    # 将data_list 写入新文件夹 "/home/wngys/lab/DeepFold/pair/train_pair_bool_90"
    f_w = open(newDir+id+".txt", "w")

    if total_num > batch_size: # 小于就忽略
        batch_num = math.floor((total_num - 1)/ batch_size) + 1
        # print(batch_num,end = ' ')
        left_num = total_num % batch_size # 没有用到?
        # print(left_num,end=' ')
        posi_num_batch = math.floor((posi_num - 1)/ batch_num) + 1
        # print(posi_num_batch,end = ' ')
        posi_left = posi_num % batch_num
        # print(posi_left, end = ' ')
        # 如果余数不为0 填充正例
        # print(batch_num-posi_left)

        if posi_num > batch_num-posi_left:
            posi_data.extend(posi_data[:(batch_num-posi_left)])
        else:
            cnt = 0
            tmp_data = copy.deepcopy(posi_data)
            while cnt < batch_num - posi_left:
                cnt += posi_num
                posi_data.extend(tmp_data)
                # print(len(tmp_data))

        nega_cusor = 0
        for i in range(batch_num):
            # data_list.extend(posi_data[i*posi_num_batch:(i+1)*posi_num_batch])
            for data in posi_data[i*posi_num_batch:(i+1)*posi_num_batch]:
                f_w.write(data)
            if i < batch_num - 1:
                # data_list.extend(nega_data[nega_cusor:nega_cusor+(batch_size-posi_num_batch)])
                for data in nega_data[nega_cusor:nega_cusor+(batch_size-posi_num_batch)]:
                    f_w.write(data)
                nega_cusor += batch_size-posi_num_batch
            else: # 最后一个分支 负例也填不满batch_size
                if batch_num - posi_left > batch_size - left_num:
                    # data_list.extend(nega_data[nega_cusor:nega_cusor+(batch_size-posi_num_batch)]) # 和前面batch处理相同，最后余出的负例不要了
                    for data in nega_data[nega_cusor:nega_cusor+(batch_size-posi_num_batch)]:
                        f_w.write(data)
                else:
                    # data_list.extend(nega_data[nega_cusor:])
                    k = 0
                    # while(k <= 10):
                        # k += len(nega_data[nega_cusor:])
                    for data in nega_data[nega_cusor:]:
                        f_w.write(data)

    else:
        # data_list.extend(posi_data)
        # data_list.extend(nega_data)
        for data in posi_data:
            f_w.write(data)
        for data in nega_data:
            f_w.write(data)
        
    f_w.close()











    