import numpy as np
import random

# node_num = 0  # 每个时刻网络中节点的总数
# # % 构建超网络
# total_hyper_e_num = 500  # 非均匀BA无标度超网络中的总的超边数
# Max_node = 10     # 每条超边内的最大节点数
# # hyper_e_num_t = 1  # t时刻非均匀BA无标度超网络中的超边数

def BA_uniform(total_hyper_e_num, new_node, k):
    # 创建k均匀BA超网络
    # new_node = k - 2   # 每一步加入的新节点
    old_node = k - new_node
    N = (total_hyper_e_num - 1) * new_node + k  # K均匀无标度超网络的总节点数

    # 构建超边矩阵和节点超度矩阵
    hyper_edge_matrix = np.zeros((total_hyper_e_num, N))
    hyper_node_degree = np.zeros((1, N))
    for i in range(k):
        hyper_edge_matrix[0, i] = 1
    for i in range(k):
        hyper_node_degree[0, i] = 1

    # 计算超度概率
    hyper_degree_frequence = hyper_node_degree / np.sum(hyper_node_degree)

    # 生成剩下的超边
    node_num = k
    for j in range(1, total_hyper_e_num):
        node_num = k + (j - 1) * new_node
        hyper_edge_matrix[j, node_num] = 1
        hyper_node_degree[0, node_num] += 1
        add_node = 0
        while add_node < old_node:
            sj = random.uniform(0, 1)
            Sum = np.cumsum(hyper_degree_frequence)
            choose_node = np.argmax(Sum >= sj)
            if hyper_edge_matrix[j, choose_node] != 1:
                hyper_edge_matrix[j, choose_node] = 1
                add_node += 1
                hyper_node_degree[0, choose_node] += 1
        hyper_degree_frequence = hyper_node_degree / np.sum(hyper_node_degree)
    
    return hyper_edge_matrix

def BA_non_uniform_new(total_hyper_e_num, new_node, Max_node):

    # % 初始先生成一个超边
    k1 = random.randint(1, Max_node)  # 生成一个[1，Max_node）随机数作为第一条超边的节点数
    # new_node = 1   # 新节点数加入的数量

    # % 非均匀BA无标度超网络的总节点数
    N = k1 + new_node * (total_hyper_e_num - 1) 
    # % 超边矩阵
    hyper_edge_matrix = np.zeros((total_hyper_e_num, N))
    # % 节点超度矩阵
    hyper_node_degree = np.zeros(N)  # 长度为n的全0数组

    for i in range(1, k1 + 1): # i从1到k（包括k），循环k次
        hyper_edge_matrix[0, i - 1] = 1
        node_num = k1
    # % 超度矩阵赋初值，初始任意k个节点的超度都为1
    for i in range(node_num): # i从0到node_num-1，循环node_num次
        hyper_node_degree[i] = 1
    # % 每次生成一条超边时重新计算超度概率
    sumhyperdegree = sum(hyper_node_degree)
    hyper_degree_frequence = hyper_node_degree / sumhyperdegree

    # % 再生成剩下的超边
    # 生成2个新节点,剩下的节点从旧节点中选择
    # 先将节点生成
    for j in range(1, total_hyper_e_num):
        nn = node_num  # nn为当前总结点数
        # 生成当前超边结点数 
        k = random.randint(new_node + 1, Max_node)  # 为了不生成独立超边，当前超边节点数需>=new_node + 1
        while(k >= node_num + new_node):  # 跳出循环条件：k - new_node >= node_num
            k = random.randint(new_node + 1, Max_node)
        old_node = k - new_node
        node_num = k1 + j * new_node
        for j1 in range(nn, node_num):
            hyper_edge_matrix[j, j1] = 1
            hyper_node_degree[j1] = 1
        # % 选择旧节点加入
        add_node = 0
        while (add_node < old_node):
            Sum = np.zeros((1, N))   # 1行N列的数组
            Sum[0, 0] = hyper_degree_frequence[0]
            sj = random.uniform(0, 1)         # 产生一个随机浮点数

            for i in range(1, N):
                Sum[0, i] = Sum[0, i - 1] + hyper_degree_frequence[i]

            if sj <= Sum[0, 1]:  # 赌轮法左开右闭
                choose_node = 1
            else:
                for i in range(2, N):
                    if sj > Sum[0, i - 1] and sj <= Sum[0, i]:
                        choose_node = i
                        break

            if hyper_edge_matrix[j, choose_node - 1] != 1:
                hyper_edge_matrix[j, choose_node - 1] = 1
                add_node = add_node + 1
                # 修改超度矩阵
                hyper_node_degree[choose_node - 1] = hyper_node_degree[choose_node - 1] + 1

        #  每次生成一条超边时,重新计算超度概率
        sumhyperdegree = sum(hyper_node_degree)
        hyper_degree_frequence = hyper_node_degree / sumhyperdegree
        # 网络构建完成
    # print(hyper_node_degree)
    # print(hyper_edge_matrix)

    return hyper_edge_matrix


def BA_non_uniform_old(total_hyper_e_num, old_node, Max_node):
    # 初始化节点数和超边矩阵
    N_0 = Max_node * (total_hyper_e_num - 1)
    hyper_edge_matrix = np.zeros((total_hyper_e_num, N_0))
    hyper_node_degree = np.zeros(N_0)

    for j in range(total_hyper_e_num):
        # 随机生成新节点数, 保证 new_nodes 和 old_node 数目有效
        new_nodes = np.random.randint(1, Max_node - old_node + 1)
        
        if j == 0:
            # 初始化第一个超边
            node_num = old_node + new_nodes
            hyper_edge_matrix[0, :node_num] = 1
            hyper_node_degree[:node_num] = 1
        else:
            # 更新节点总数，并添加新节点
            node_num += new_nodes
            hyper_edge_matrix[j, node_num - new_nodes:node_num] = 1
            hyper_node_degree[node_num - new_nodes:node_num] = 1

            add_node = 0
            used_nodes = set(range(node_num - new_nodes, node_num))  # 已使用的新节点
            while add_node < old_node:
                # 计算节点度的概率分布
                sum_hyper_degree = np.sum(hyper_node_degree[:node_num])
                hyper_degree_frequence = hyper_node_degree[:node_num] / sum_hyper_degree
                cumulative_prob = np.cumsum(hyper_degree_frequence)

                # 通过轮盘赌选择一个旧节点
                rand_value = random.uniform(0, cumulative_prob[-1])
                selected_node = np.searchsorted(cumulative_prob, rand_value)

                # 确保所选节点未被选过
                if selected_node not in used_nodes:
                    hyper_edge_matrix[j, selected_node] = 1
                    hyper_node_degree[selected_node] += 1
                    used_nodes.add(selected_node)
                    add_node += 1

    # 删除没有连接的节点
    non_zero_columns = (hyper_edge_matrix.sum(axis=0) != 0)
    hyper_edge_matrix = hyper_edge_matrix[:, non_zero_columns]
    hyper_node_degree = hyper_node_degree[non_zero_columns]

    return hyper_edge_matrix


def BA_non_uniform(total_hyper_e_num, Max_node):
    # 初始化节点数和超边矩阵
    N_0 = Max_node * total_hyper_e_num
    hyper_edge_matrix = np.zeros((total_hyper_e_num, N_0))
    hyper_node_degree = np.zeros(N_0)

    for j in range(total_hyper_e_num):
        # 随机生成新节点数和旧节点数
        new_nodes = np.random.randint(1, Max_node)
        old_node = np.random.randint(1, Max_node - new_nodes + 1)
        
        if j == 0:
            # 初始化第一个超边
            node_num = old_node + new_nodes
            for i in range(node_num):
                hyper_edge_matrix[0, i] = 1
                hyper_node_degree[i] = 1
        else:
            # 更新节点总数，并添加新节点
            node_num += new_nodes
            for j1 in range(node_num - new_nodes, node_num):
                hyper_edge_matrix[j, j1] = 1
                hyper_node_degree[j1] = 1

            add_node = 0
            used_nodes = set(range(node_num - new_nodes, node_num))
            while add_node < old_node:
                # 重新计算节点度的概率分布
                sum_hyper_degree = np.sum(hyper_node_degree[:node_num])
                hyper_degree_frequence = hyper_node_degree[:node_num] / sum_hyper_degree
                cumulative_prob = np.cumsum(hyper_degree_frequence)

                # 通过轮盘赌选择一个旧节点
                rand_value = random.uniform(0, cumulative_prob[-1])
                selected_node = np.searchsorted(cumulative_prob, rand_value)

                # 确保所选节点是有效的并未被选过
                if selected_node not in used_nodes:
                    hyper_edge_matrix[j, selected_node] = 1
                    hyper_node_degree[selected_node] += 1
                    used_nodes.add(selected_node)
                    add_node += 1

    # 删除未使用的列
    non_zero_columns = (hyper_edge_matrix.sum(axis=0) != 0)
    hyper_edge_matrix = hyper_edge_matrix[:, non_zero_columns]
    hyper_node_degree = hyper_node_degree[non_zero_columns]

    return hyper_edge_matrix