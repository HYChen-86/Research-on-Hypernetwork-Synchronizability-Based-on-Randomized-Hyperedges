import numpy as np

# # 参数设置
# max_node = 10
# M = 20  # 超边数量（可以修改为其他值）
# N_0 = M * max_node

def WS_non_uniform(total_hyper_e_num, Max_node):
    # 参数设置
    M = total_hyper_e_num
    max_node = Max_node
    N_0 = M * max_node
    p = 0.3  # 断边重连概率

    # 初始化超网络矩阵
    hyper_network_matrix = np.zeros((M, N_0))
    node_number = 0

    # 构建超网络矩阵(添加公共节点)
    for i in range(M-2):
        old_node = np.random.randint(1, int(max_node/3))
        hyper_network_matrix[i, node_number: node_number+old_node] = 1
        hyper_network_matrix[i + 1, node_number: node_number+old_node] = 1
        hyper_network_matrix[i + 2, node_number: node_number+old_node] = 1
        node_number += old_node

    i = M - 2
    old_node = np.random.randint(1, int(max_node/3))
    hyper_network_matrix[i, node_number: node_number+old_node] = 1
    hyper_network_matrix[i + 1, node_number: node_number+old_node] = 1
    hyper_network_matrix[0, node_number: node_number+old_node] = 1
    node_number += old_node

    i = M - 1
    old_node = np.random.randint(1, int(max_node/3))
    hyper_network_matrix[i, node_number: node_number+old_node] = 1
    hyper_network_matrix[0, node_number: node_number+old_node] = 1
    hyper_network_matrix[1, node_number: node_number+old_node] = 1
    node_number += old_node

    # 构建超网络矩阵(添加非公共节点)
    for j in range(M):
        temp_count = np.count_nonzero(hyper_network_matrix[j, :])
        if max_node - temp_count > 1:
            new_node = np.random.randint(1, max_node - temp_count)
        else:
            new_node = 1
        hyper_network_matrix[j, node_number: node_number + new_node] = 1
        node_number += new_node

    # 断超边重连
    for k in range(int(M * p)):
        target = np.random.randint(M)
        # 处理一般情况
        if target < M-2:
            # 取掉旧超边
            for n in range(N_0):
                if hyper_network_matrix[target, n] * hyper_network_matrix[target + 2, n] == 1:
                    hyper_network_matrix[target + 2, n] = 0
            # 重连新超边
            new_target = target
            while new_target == target or new_target == target + 1 or new_target == target - 1:
                new_target = np.random.randint(M)
            if new_target < M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[new_target + 2, m] == 1:
                        hyper_network_matrix[target + 2, m] = 1
            elif new_target == M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[0, m] == 1:
                        hyper_network_matrix[target + 2, m] = 1
            elif new_target == M-1:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[1, m] == 1:
                        hyper_network_matrix[target + 2, m] = 1
        # 处理断开超边为倒数第二条超边的情况
        elif target == M-2:
            for n in range(N_0):
                if hyper_network_matrix[target, n] * hyper_network_matrix[0, n] == 1:
                    hyper_network_matrix[0, n] = 0
            # 重连新超边
            new_target = target
            while new_target == target or new_target == target + 1 or new_target == target - 1:
                new_target = np.random.randint(M)
            if new_target < M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[new_target + 2, m] == 1:
                        hyper_network_matrix[0, m] = 1
            elif new_target == M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[0, m] == 1:
                        hyper_network_matrix[0, m] = 1
            elif new_target == M-1:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[1, m] == 1:
                        hyper_network_matrix[0, m] = 1
        # 处理断开超边为倒数第一条超边的情况                
        elif target == M-1:
            for n in range(N_0):
                if hyper_network_matrix[target, n] * hyper_network_matrix[1, n] == 1:
                    hyper_network_matrix[1, n] = 0
            # 重连新超边
            new_target = target
            while new_target == target or new_target == target + 1 or new_target == target - 1:
                new_target = np.random.randint(M)
            if new_target < M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[new_target + 2, m] == 1:
                        hyper_network_matrix[1, m] = 1
            elif new_target == M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[0, m] == 1:
                        hyper_network_matrix[1, m] = 1
            elif new_target == M-1:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[1, m] == 1:
                        hyper_network_matrix[1, m] = 1

    # 删除超网络矩阵中全为 0 的列
    non_zero_columns = np.any(hyper_network_matrix, axis=0)
    hyper_network_matrix = hyper_network_matrix[:, non_zero_columns]

    return hyper_network_matrix

def WS_non_uniform_old(total_hyper_e_num, Max_node, old_node):
    # 参数设置
    M = total_hyper_e_num
    max_node = Max_node
    N_0 = M * max_node
    p = 0.3  # 断边重连概率
    if int(old_node/3) >0:
        old_node = int(old_node/3)
    else:
        old_node = 1

    # 初始化超网络矩阵
    hyper_network_matrix = np.zeros((M, N_0))
    node_number = 0

    # 构建超网络矩阵(添加公共节点)
    for i in range(M-2):
        # old_node = np.random.randint(1, int(max_node/3))
        hyper_network_matrix[i, node_number: node_number+old_node] = 1
        hyper_network_matrix[i + 1, node_number: node_number+old_node] = 1
        hyper_network_matrix[i + 2, node_number: node_number+old_node] = 1
        node_number += old_node

    i = M - 2
    # old_node = np.random.randint(1, int(max_node/3))
    hyper_network_matrix[i, node_number: node_number+old_node] = 1
    hyper_network_matrix[i + 1, node_number: node_number+old_node] = 1
    hyper_network_matrix[0, node_number: node_number+old_node] = 1
    node_number += old_node

    i = M - 1
    # old_node = np.random.randint(1, int(max_node/3))
    hyper_network_matrix[i, node_number: node_number+old_node] = 1
    hyper_network_matrix[0, node_number: node_number+old_node] = 1
    hyper_network_matrix[1, node_number: node_number+old_node] = 1
    node_number += old_node

    # 构建超网络矩阵(添加非公共节点)
    for j in range(M):
        temp_count = np.count_nonzero(hyper_network_matrix[j, :])
        if max_node - temp_count > 1:
            new_node = np.random.randint(1, max_node - temp_count)
        else:
            new_node = 1
        hyper_network_matrix[j, node_number: node_number + new_node] = 1
        node_number += new_node

    # 断超边重连
    for k in range(int(M * p)):
        target = np.random.randint(M)
        # 处理一般情况
        if target < M-2:
            # 取掉旧超边
            for n in range(N_0):
                if hyper_network_matrix[target, n] * hyper_network_matrix[target + 2, n] == 1:
                    hyper_network_matrix[target + 2, n] = 0
            # 重连新超边
            new_target = target
            while new_target == target or new_target == target + 1 or new_target == target - 1:
                new_target = np.random.randint(M)
            if new_target < M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[new_target + 2, m] == 1:
                        hyper_network_matrix[target + 2, m] = 1
            elif new_target == M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[0, m] == 1:
                        hyper_network_matrix[target + 2, m] = 1
            elif new_target == M-1:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[1, m] == 1:
                        hyper_network_matrix[target + 2, m] = 1
        # 处理断开超边为倒数第二条超边的情况
        elif target == M-2:
            for n in range(N_0):
                if hyper_network_matrix[target, n] * hyper_network_matrix[0, n] == 1:
                    hyper_network_matrix[0, n] = 0
            # 重连新超边
            new_target = target
            while new_target == target or new_target == target + 1 or new_target == target - 1:
                new_target = np.random.randint(M)
            if new_target < M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[new_target + 2, m] == 1:
                        hyper_network_matrix[0, m] = 1
            elif new_target == M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[0, m] == 1:
                        hyper_network_matrix[0, m] = 1
            elif new_target == M-1:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[1, m] == 1:
                        hyper_network_matrix[0, m] = 1
        # 处理断开超边为倒数第一条超边的情况                
        elif target == M-1:
            for n in range(N_0):
                if hyper_network_matrix[target, n] * hyper_network_matrix[1, n] == 1:
                    hyper_network_matrix[1, n] = 0
            # 重连新超边
            new_target = target
            while new_target == target or new_target == target + 1 or new_target == target - 1:
                new_target = np.random.randint(M)
            if new_target < M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[new_target + 2, m] == 1:
                        hyper_network_matrix[1, m] = 1
            elif new_target == M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[0, m] == 1:
                        hyper_network_matrix[1, m] = 1
            elif new_target == M-1:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[1, m] == 1:
                        hyper_network_matrix[1, m] = 1

    # 删除超网络矩阵中全为 0 的列
    non_zero_columns = np.any(hyper_network_matrix, axis=0)
    hyper_network_matrix = hyper_network_matrix[:, non_zero_columns]

    return hyper_network_matrix

def WS_non_uniform_new(total_hyper_e_num, Max_node, new_node):
    # 参数设置
    M = total_hyper_e_num
    max_node = Max_node
    N_0 = M * max_node
    p = 0.3  # 断边重连概率

    # 初始化超网络矩阵
    hyper_network_matrix = np.zeros((M, N_0))
    node_number = 0

    # 构建超网络矩阵(添加公共节点)
    for i in range(M-2):
        old_node = np.random.randint(1, int(max_node/3))
        hyper_network_matrix[i, node_number: node_number+old_node] = 1
        hyper_network_matrix[i + 1, node_number: node_number+old_node] = 1
        hyper_network_matrix[i + 2, node_number: node_number+old_node] = 1
        node_number += old_node

    i = M - 2
    old_node = np.random.randint(1, int(max_node/3))
    hyper_network_matrix[i, node_number: node_number+old_node] = 1
    hyper_network_matrix[i + 1, node_number: node_number+old_node] = 1
    hyper_network_matrix[0, node_number: node_number+old_node] = 1
    node_number += old_node

    i = M - 1
    old_node = np.random.randint(1, int(max_node/3))
    hyper_network_matrix[i, node_number: node_number+old_node] = 1
    hyper_network_matrix[0, node_number: node_number+old_node] = 1
    hyper_network_matrix[1, node_number: node_number+old_node] = 1
    node_number += old_node

    # 构建超网络矩阵(添加非公共节点)
    for j in range(M):
        temp_count = np.count_nonzero(hyper_network_matrix[j, :])
        # new_node = np.random.randint(1, max_node - temp_count)
        hyper_network_matrix[j, node_number: node_number + new_node] = 1
        node_number += new_node

    # 断超边重连
    for k in range(int(M * p)):
        target = np.random.randint(M)
        # 处理一般情况
        if target < M-2:
            # 取掉旧超边
            for n in range(N_0):
                if hyper_network_matrix[target, n] * hyper_network_matrix[target + 2, n] == 1:
                    hyper_network_matrix[target + 2, n] = 0
            # 重连新超边
            new_target = target
            while new_target == target or new_target == target + 1 or new_target == target - 1:
                new_target = np.random.randint(M)
            if new_target < M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[new_target + 2, m] == 1:
                        hyper_network_matrix[target + 2, m] = 1
            elif new_target == M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[0, m] == 1:
                        hyper_network_matrix[target + 2, m] = 1
            elif new_target == M-1:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[1, m] == 1:
                        hyper_network_matrix[target + 2, m] = 1
        # 处理断开超边为倒数第二条超边的情况
        elif target == M-2:
            for n in range(N_0):
                if hyper_network_matrix[target, n] * hyper_network_matrix[0, n] == 1:
                    hyper_network_matrix[0, n] = 0
            # 重连新超边
            new_target = target
            while new_target == target or new_target == target + 1 or new_target == target - 1:
                new_target = np.random.randint(M)
            if new_target < M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[new_target + 2, m] == 1:
                        hyper_network_matrix[0, m] = 1
            elif new_target == M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[0, m] == 1:
                        hyper_network_matrix[0, m] = 1
            elif new_target == M-1:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[1, m] == 1:
                        hyper_network_matrix[0, m] = 1
        # 处理断开超边为倒数第一条超边的情况                
        elif target == M-1:
            for n in range(N_0):
                if hyper_network_matrix[target, n] * hyper_network_matrix[1, n] == 1:
                    hyper_network_matrix[1, n] = 0
            # 重连新超边
            new_target = target
            while new_target == target or new_target == target + 1 or new_target == target - 1:
                new_target = np.random.randint(M)
            if new_target < M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[new_target + 2, m] == 1:
                        hyper_network_matrix[1, m] = 1
            elif new_target == M-2:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[0, m] == 1:
                        hyper_network_matrix[1, m] = 1
            elif new_target == M-1:
                for m in range(N_0):
                    if hyper_network_matrix[new_target, m] * hyper_network_matrix[1, m] == 1:
                        hyper_network_matrix[1, m] = 1

    # 删除超网络矩阵中全为 0 的列
    non_zero_columns = np.any(hyper_network_matrix, axis=0)
    hyper_network_matrix = hyper_network_matrix[:, non_zero_columns]

    return hyper_network_matrix