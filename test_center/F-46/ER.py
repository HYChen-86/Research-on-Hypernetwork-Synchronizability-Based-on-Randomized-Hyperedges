import numpy as np

def ER_non_uniform(total_hyper_e_num, Max_node):
    # 参数设置
    # max_node = 10
    # M = 20  # 超边数量（可以修改为其他值）
    max_node = Max_node
    M = total_hyper_e_num  # 超边数量（可以修改为其他值）
    # N_1 = int(np.sqrt(2 * M) - 1)  # 初始节点池大小
    N_1 = int(0.4 * M * max_node)  # 初始节点池大小
    # old_node = np.random.randint(1, min(N_1 + 1, max_node))  # 每条超边的公共节点数
    # new_node = np.random.randint(1, max_node - old_node)  # 每条超边的非公共节点数
    N_0 = M * max_node  # 总节点池大小

    # 初始化超网络矩阵
    hyper_network_matrix = np.zeros((M, N_0))
    node_number = N_1  # 记录当前非公共节点部分的起始索引

    # 确保初始节点池中的每个节点至少出现在两个超边中
    used_count = np.zeros(N_1, dtype=int)  # 记录每个初始节点池节点的使用次数

    # 构建超网络矩阵（确保每个初始节点池中的节点至少使用两次）
    for i in range(M):
        old_node = np.random.randint(1, min(N_1 + 1, max_node))
        # 随机选取 old_node 个节点，确保范围在 [0, N_1-1]
        selected_old_nodes = np.random.choice(range(N_1), size=old_node, replace=False)
        used_count[selected_old_nodes] += 1
        hyper_network_matrix[i, selected_old_nodes] = 1  # 将选取的节点加入第 i 条超边

    # 补充：确保初始节点池中的每个节点至少被选中两次
    for node_idx in range(N_1):
        while used_count[node_idx] < 2:
            # 找到未完全使用的节点并随机分配到一个超边
            target_edge = np.random.randint(0, M)  # 随机选择一个超边
            if hyper_network_matrix[target_edge, node_idx] == 0:  # 确保节点未被包含
                hyper_network_matrix[target_edge, node_idx] = 1
                used_count[node_idx] += 1

    # 输出调试信息
    # print("初始节点池大小 (N_1):", N_1)
    # print("每个节点的使用次数:", used_count)
    # print("超网络矩阵初始形状:", hyper_network_matrix.shape)

    # 构建网络矩阵（补充非公共节点部分）
    for j in range(M):
        temp_count = np.count_nonzero(hyper_network_matrix[j, :])
        if max_node - temp_count > 0:
            new_node = np.random.randint(0, max_node - temp_count)
        else:
            new_node = 0
        # new_node = np.random.randint(0, int(0.1*M))
        hyper_network_matrix[j, node_number: node_number + new_node] = 1
        node_number += new_node

    # 删除超网络矩阵中全为 0 的列
    non_zero_columns = np.any(hyper_network_matrix, axis=0)
    hyper_network_matrix = hyper_network_matrix[:, non_zero_columns]

    # 输出信息
    # print("超网络矩阵最终形状:", hyper_network_matrix.shape)

    return hyper_network_matrix

def ER_non_uniform_old(total_hyper_e_num, Max_node, old_node):
    # 参数设置
    max_node = Max_node
    M = total_hyper_e_num  # 超边数量（可以修改为其他值）
    # N_1 = int(np.sqrt(2 * M) - 1)  # 初始节点池大小
    N_1 = int(0.4 * M * max_node)  # 初始节点池大小
    N_0 = M * max_node  # 总节点池大小

    # 初始化超网络矩阵
    hyper_network_matrix = np.zeros((M, N_0))
    node_number = N_1  # 记录当前非公共节点部分的起始索引

    # 确保初始节点池中的每个节点至少出现在两个超边中
    used_count = np.zeros(N_1, dtype=int)  # 记录每个初始节点池节点的使用次数

    # 构建超网络矩阵（确保每个初始节点池中的节点至少使用两次）
    for i in range(M):
        # old_node = np.random.randint(1, min(N_1 + 1, max_node))
        old_node = old_node
        # 随机选取 old_node 个节点，确保范围在 [0, N_1-1]
        selected_old_nodes = np.random.choice(range(N_1), size=old_node, replace=False)
        used_count[selected_old_nodes] += 1
        hyper_network_matrix[i, selected_old_nodes] = 1  # 将选取的节点加入第 i 条超边

    # 补充：确保初始节点池中的每个节点至少被选中两次
    for node_idx in range(N_1):
        while used_count[node_idx] < 2:
            # 找到未完全使用的节点并随机分配到一个超边
            target_edge = np.random.randint(0, M)  # 随机选择一个超边
            if hyper_network_matrix[target_edge, node_idx] == 0:  # 确保节点未被包含
                hyper_network_matrix[target_edge, node_idx] = 1
                used_count[node_idx] += 1

    # 构建网络矩阵（补充非公共节点部分）
    for j in range(M):
        temp_count = np.count_nonzero(hyper_network_matrix[j, :])
        # new_node = np.random.randint(1, max_node - temp_count)
        new_node = np.random.randint(0, int(0.1*M))
        hyper_network_matrix[j, node_number: node_number + new_node] = 1
        node_number += new_node

    # 删除超网络矩阵中全为 0 的列
    non_zero_columns = np.any(hyper_network_matrix, axis=0)
    hyper_network_matrix = hyper_network_matrix[:, non_zero_columns]

    return hyper_network_matrix

def ER_non_uniform_new(total_hyper_e_num, Max_node, new_node):
    # 参数设置
    max_node = Max_node
    M = total_hyper_e_num  # 超边数量（可以修改为其他值）
    # N_1 = int(np.sqrt(2 * M) - 1)  # 初始节点池大小
    N_1 = int(0.4 * M * max_node)  # 初始节点池大小
    N_0 = M * max_node  # 总节点池大小

    # 初始化超网络矩阵
    hyper_network_matrix = np.zeros((M, N_0))
    node_number = N_1  # 记录当前非公共节点部分的起始索引

    # 确保初始节点池中的每个节点至少出现在两个超边中
    used_count = np.zeros(N_1, dtype=int)  # 记录每个初始节点池节点的使用次数

    # 构建超网络矩阵（确保每个初始节点池中的节点至少使用两次）
    for i in range(M):
        old_node = np.random.randint(1, min(N_1 + 1, max_node))
        # old_node = old_node
        # 随机选取 old_node 个节点，确保范围在 [0, N_1-1]
        selected_old_nodes = np.random.choice(range(N_1), size=old_node, replace=False)
        used_count[selected_old_nodes] += 1
        hyper_network_matrix[i, selected_old_nodes] = 1  # 将选取的节点加入第 i 条超边

    # 补充：确保初始节点池中的每个节点至少被选中两次
    for node_idx in range(N_1):
        while used_count[node_idx] < 2:
            # 找到未完全使用的节点并随机分配到一个超边
            target_edge = np.random.randint(0, M)  # 随机选择一个超边
            if hyper_network_matrix[target_edge, node_idx] == 0:  # 确保节点未被包含
                hyper_network_matrix[target_edge, node_idx] = 1
                used_count[node_idx] += 1

    # 构建网络矩阵（补充非公共节点部分）
    for j in range(M):
        temp_count = np.count_nonzero(hyper_network_matrix[j, :])
        # new_node = np.random.randint(1, max_node - temp_count)
        # new_node = np.random.randint(0, int(0.1*M))
        new_node = new_node
        hyper_network_matrix[j, node_number: node_number + new_node] = 1
        node_number += new_node

    # 删除超网络矩阵中全为 0 的列
    non_zero_columns = np.any(hyper_network_matrix, axis=0)
    hyper_network_matrix = hyper_network_matrix[:, non_zero_columns]

    return hyper_network_matrix
