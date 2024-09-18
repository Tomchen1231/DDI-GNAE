from torch_geometric.utils import train_test_split_edges
from DrugData_load import MyDrugDataset
import pandas as pd
import numpy as np

dataset = MyDrugDataset('the_data', 'tcm_more')
# dataset = MyDrugDataset('the_data', 'drug_four_lowdim_0.98')
data = dataset[0]

data = train_test_split_edges(data, val_ratio=0, test_ratio=1.0)

# 得到正负样本索引
test_pos, test_neg = data.test_pos_edge_index, data.test_neg_edge_index  # 2*45199为什么少了？

def get_drugname():
    # 拿到药物的index和id
    drug_list_csv = pd.read_csv("the_data/tcm_data/herb_three_attri_first_indication.csv", encoding="utf-8")

    drug_index = np.array(np.vstack((drug_list_csv['index'], drug_list_csv['TCMBank_ID'])))
    drug_index = drug_index.T
    # 把index和id组成字典
    DB_index = drug_index[:, 0]  # 取所有行的第0个数据
    DB_ID = drug_index[:, 1]  # [:,1] 取所有行的第1个数据

    DB_name = drug_index[:, 2]

    dic = dict(zip(DB_index, DB_ID))  # 组成字典

    # 正样本 标签和索引
    pos_a_index = test_pos[0].tolist()
    pos_b_index = test_pos[1].tolist()
    pos_pairs = []

    pos_a_name = [dic[item] for item in pos_a_index.numpy()]
    pos_b_name = [dic[item] for item in pos_b_index.numpy()]
    for i in range(len(pos_a_index)):
        pos_pairs.append([pos_a_name[i], pos_b_name[i], 1])
    pos_pairs = pd.DataFrame(pos_pairs, columns=['namea', 'nameb', 'label'])
    pos_pairs.to_csv('the_data/pos_pairs.csv', index=False, header=False)


    for i in range(len(pos_a_index)):
        pos_pairs.append([pos_a_index[i], pos_b_index[i], 1])
    pos_pairs = pd.DataFrame(pos_pairs, columns=['namea', 'nameb', 'label'])
    pos_pairs.to_csv('the_data/tcm_data/pos_pairs.csv', index=False, header=False)

    # # 负样本
    # neg_a_index = test_neg[0].tolist()
    # neg_b_index = test_neg[1].tolist()
    # neg_pairs = []
    #
    # # neg_a_name = [dic[item] for item in neg_a_index.numpy()]
    # # neg_b_name = [dic[item] for item in neg_b_index.numpy()]
    # for i in range(len(neg_a_index)):
    #     neg_pairs.append([neg_a_index[i], neg_b_index[i], 0])
    # neg_pairs = pd.DataFrame(neg_pairs, columns=['namea', 'nameb', 'label'])
    #
    # neg_pairs.to_csv('the_data/tcm_data/neg_pairs.csv', index=False, header=False)

    # # # 边，两个药物的id
    # link_csv = pd.read_csv("the_data/all_move_point_35W.csv", encoding="utf-8")
    # DB_ID_a = link_csv['ida']
    # DB_ID_b = link_csv['idb']
    #
    # index_a = [dic[item] for item in DB_ID_a]  # 取所有行的第1个数据   positive1是mat_ddi第一个药物的药物索引
    # index_b = [dic[item] for item in DB_ID_b]  # positive2是mat_ddi第二个药物的药物索引
    # drug_pairs = np.array(np.vstack((index_a, index_b)))  # 第一个药物索引，第二个药物索引
    # drug_pairs = drug_pairs.T
    #
    # drug_pairs = pd.DataFrame(drug_pairs, columns=['a', 'b'])
    # drug_pairs.to_csv('edge/edge_pos.txt', sep=' ', index=False, header=False)

    # np.savetxt(fname="edge/edge_pos.txt", X=drug_pairs, fmt="%d", delimiter=" ")


get_drugname()

print(data)
