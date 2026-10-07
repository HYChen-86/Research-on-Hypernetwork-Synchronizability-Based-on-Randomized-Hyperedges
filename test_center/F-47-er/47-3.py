import numpy as np

# 构建三维邻接矩阵Supra-Adjacent
def Supra_Adjacent(hyper_network):
    M = hyper_network.shape[0]
    N = hyper_network.shape[1]
    structural_matrix = np.zeros((M, N, N))
    for k in range(M):
        for j in range(N):
            for i in range(j, N):
                if hyper_network[k][j] == hyper_network[k][i] == 1 and i != j:
                    structural_matrix[k][i][j] = 1
                    structural_matrix[k][j][i] = 1

    return structural_matrix

# 构建超边内Laplacian矩阵L_1
def Laplacian_inner(matrix, C):
    M = matrix.shape[0]
    N = matrix.shape[1]
    L_1 = np.zeros((N, N))
    for k in range(M):
        L = np.zeros_like(L_1)
        count_nonzero = 0
        for i in range(N):
            count_nonzero = np.count_nonzero(matrix[k, i, :])
            if count_nonzero != 0:
                L[i, :] = matrix[k, i, :] / count_nonzero
                L[i][i] = -np.sum(matrix[k, i, :]) / count_nonzero
        L_1 += L
    return L_1 * C

# 计算特征值比R
def Eigenvalue_ratio(adjacency_matrix):
    # 计算邻接矩阵的特征值
    eigenvalues, _ = np.linalg.eig(adjacency_matrix)
    # 处理浮点数误差，确保结果为实数并且接近零的数值设置为零
    eigenvalues = np.real(eigenvalues)  # 取实部
    eigenvalues[np.isclose(eigenvalues, 0)] = 0  # 将接近零的值替换为零
    
    # 找到特征值中的最大值和第二大值
    max_eigenvalue = max(eigenvalues)
    sorted_eigenvalues = np.sort(eigenvalues)
    second_max_eigenvalue = sorted_eigenvalues[1]

    # 计算特征值比 R
    # R = max_eigenvalue / second_max_eigenvalue

    # print("特征值比 R:", R)
    return sorted_eigenvalues, second_max_eigenvalue, eigenvalues

#  计算各层特征值序列
def Eigenvalue_ratio_mutiorder(adjacency_matrix):
    Lambda_2_order = np.zeros(M)
    Lambda_order = np.zeros((M, N))
    for k in range(M):
        matrix = np.zeros((N, N))
        matrix[:, :] = adjacency_matrix[k, :, :]
        # 删除全为0的列
        non_zero_columns = (matrix.sum(axis=0) != 0)
        matrix = matrix[:, non_zero_columns]
        # 删除全为0的行
        non_zero_rows = (matrix.sum(axis=1) != 0)
        matrix = matrix[non_zero_rows]
        # 转换为层Laplacian
        count_nonzero = 0
        for i in range(matrix.shape[0]):
            count_nonzero = np.count_nonzero(matrix[i, :])
            if count_nonzero > 0:
                matrix[i, :] = matrix[i, :] / count_nonzero
                matrix[i][i] = -np.sum(matrix[i, :])
        # 计算特征值
        sorted_eigenvalues, second_max_eigenvalue, eigenvalues = Eigenvalue_ratio(adjacency_matrix = -matrix * C_1)
        Lambda_2_order[k] = second_max_eigenvalue
        # Lambda_order[k, :] = eigenvalues
    return Lambda_2_order, Lambda_order, matrix

import threading
import time
def run_with_timeout(func, timeout, *args, **kwargs):
    """
    在指定的超时时间内运行函数；如果超时则重新运行。
    """
    while True:
        # 创建一个线程来执行函数
        result_container = {}

        def wrapper():
            result_container['result'] = func(*args, **kwargs)

        thread = threading.Thread(target=wrapper)
        thread.start()
        thread.join(timeout)  # 等待执行指定时间

        # 检查线程是否完成
        if thread.is_alive():
            print("超时，重新运行函数...")
            continue  # 如果超时未完成，则重新运行
        else:
            return result_container['result']

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

# 构建非均匀ER超网络
total_hyper_e_num = 40
# for total_hyper_e_num in range(20, 140, 20):
node_number = 5
    # for node_number in range(1, 7, 1):
        # max_node = 10
for max_node in range(10, 60, 5):
    for time in range(20):
    # hyper_adjacent_matrix = BA_non_uniform_new(total_hyper_e_num=total_hyper_e_num, new_node=node_number, Max_node=max_node)  # 构建网络
    # hyper_adjacent_matrix = BA_non_uniform_old(total_hyper_e_num=total_hyper_e_num, old_node=node_number, Max_node=max_node)  # 构建网络


        # # 调用函数并设置超时
        # hyper_adjacent_matrix = run_with_timeout(
        #     BA_non_uniform_old, 30, total_hyper_e_num=total_hyper_e_num, old_node=node_number, Max_node=max_node
        # )
        # hyper_adjacent_matrix = ER_non_uniform(total_hyper_e_num=total_hyper_e_num, Max_node=max_node, target_old=node_number)
        hyper_adjacent_matrix = ER_non_uniform_new(total_hyper_e_num=total_hyper_e_num, Max_node=max_node, new_node=node_number)
        # 从CSV文件中加载矩阵
        # filename = f'network_{total_hyper_e_num}.csv'
        # hyper_adjacent_matrix = np.loadtxt(filename, delimiter=',')
        M = hyper_adjacent_matrix.shape[0]
        N = hyper_adjacent_matrix.shape[1]
        a = 0  # 统一混沌系统参数
        C_1 = 100  # 超边内耦合强度
        print("网络构建完成")
        # print(hyper_adjacent_matrix)
        
        # # 参数设置
        # x0 = np.array([40.0, 40.0, 20.0])
        # dt = 0.01  # 减小时间步长以提高数值稳定性
        # num_steps = 10000
        
        # 构建示例的拉普拉斯矩阵
        N = hyper_adjacent_matrix.shape[1]
        Supra_Adjacent_matrix = Supra_Adjacent(hyper_network = hyper_adjacent_matrix)
        L_1 = Laplacian_inner(matrix=Supra_Adjacent_matrix, C=C_1)
        print("Laplacian_Matrix构建完成")
        
        # 计算特征值
        sorted_eigenvalues, second_max_eigenvalue, eigenvalues = Eigenvalue_ratio(adjacency_matrix = -L_1)
        print("特征值序列为:", sorted_eigenvalues)
        print("Lambda_2=", second_max_eigenvalue)
        
        # 计算平均超边规模
        # Calculating the sum of each row in hyper_adjacent_matrix
        row_sums = np.sum(hyper_adjacent_matrix, axis=1)

        # Calculating the average of these row sums
        average_row_sum = np.mean(row_sums)


        with open(f'{total_hyper_e_num}_{node_number}_{max_node}.txt', 'a') as file:
                file.write(str(M) + ',' + str(second_max_eigenvalue) + ',' + str(average_row_sum) + '\n')  # 在新行追加结果