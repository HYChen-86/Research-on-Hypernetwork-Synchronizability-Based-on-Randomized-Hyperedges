import numpy as np
import random
import matplotlib.pyplot as plt

total_hyper_e_num = 20
Max_node = 10
target_old = 2

def ER_non_uniform(total_hyper_e_num, Max_node, target_old):
    # total_hyper_e_num = 20
    # Max_node = 10
    # target_old = 2
    M = total_hyper_e_num
    N_0 = Max_node * total_hyper_e_num
    
    hyper_edge_matrix = np.zeros((M, N_0))
    
    for i in range(total_hyper_e_num):
        # 为每个超边随机生成节点数量（至少一个）
        node_count = np.random.randint(1, int(Max_node/2))
    
        # 从可用节点中随机选择节点并加入超边
        selected_nodes = np.random.choice(
            range(int(N_0/8)), size=node_count, replace=False
        )
    
        # 更新超边矩阵
        hyper_edge_matrix[i, selected_nodes] = 1
        
    for j in range(M):
        for k in range(j+1, M):
            count_old = 0
            for s in range(N_0):
                if hyper_edge_matrix[j, s] * hyper_edge_matrix[k, s] == 1:
                    count_old += 1
                    if count_old > target_old:
                        hyper_edge_matrix[k, s] = 0
            if count_old < target_old:
                hyper_edge_matrix[j, int(N_0/8):target_old-count_old] = 1
                hyper_edge_matrix[k, int(N_0/8):target_old-count_old] = 1
    
    count_new = int(N_0/2)
    for t in range(M):
        temp = np.random.randint(int(Max_node/2))
        hyper_edge_matrix[j, count_new:temp] = 1
        count_new += temp
    
    # 删除未使用的列
    non_zero_columns = (hyper_edge_matrix.sum(axis=0) != 0)
    hyper_edge_matrix = hyper_edge_matrix[:, non_zero_columns]

    return hyper_edge_matrix

# 示例用法
total_hyper_e_num = 100
Max_node = 10
target_old = 2
hyper_edge_matrix = ER_non_uniform(total_hyper_e_num, Max_node, target_old)
print(hyper_edge_matrix)

# 绘制节点超度分布
N = hyper_edge_matrix.shape[1]
M = hyper_edge_matrix.shape[0]
hyper_node_degree = np.zeros(N)
for node in range(N):
    hyper_node_degree[node] = np.sum(hyper_edge_matrix[:, node])

x = range(1, len(hyper_node_degree) + 1)  # 生成X轴序列，从1到最大度
y = [z / float(sum(hyper_node_degree)) for z in hyper_node_degree]  # 将频次转化为频率

plt.figure(figsize=(6, 6), dpi=150)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号
plt.xlabel("Degree", size=14)  # Degree
plt.ylabel("Frequency", size=14)  # Frequency
plt.xticks(fontproperties='Times New Roman', size=13)
plt.yticks(fontproperties='Times New Roman', size=13)
plt.loglog(x, y, '.', label="Node Degree Distribution")

import matplotlib.pyplot as plt

hyper_node_degree = np.zeros(N)
for i in range(N):
    count = 0
    for j in range(M):
        # if hyper_networks_matrix[j][i] == 1 and higherOrder_structure[j][0] == 4:
        if hyper_edge_matrix[j][i] == 1:
            count += 1
    hyper_node_degree[i] = count

#计算所有超度的总和
hyper_node_degree_sum = sum(hyper_node_degree)
#将超度从大到小排序
hyper_node_degree1 = sorted(hyper_node_degree,reverse = True)
unique_data = np.unique(hyper_node_degree1)   #去除数组中重复的数字，并从小到大排序后进行输出1
maxdegree = unique_data[-1]
maxdegree = int(maxdegree)  #获得超度的最大值
uu = maxdegree + 10
resdata = []
for o in range(uu):
    resdata.append(hyper_node_degree1.count(o))   #统计数字出现的次数
#画超度分布图
plt.plot(range(uu),resdata/hyper_node_degree_sum,color = 'darkblue')
plt.xlabel('node')
plt.ylabel('hyper_node_degree')
plt.show()