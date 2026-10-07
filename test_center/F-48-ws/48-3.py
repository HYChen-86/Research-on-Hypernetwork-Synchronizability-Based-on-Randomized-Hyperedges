import numpy as np
from WS import WS_non_uniform, WS_non_uniform_new, WS_non_uniform_old

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
        hyper_adjacent_matrix = WS_non_uniform_new(total_hyper_e_num=total_hyper_e_num, Max_node=max_node, new_node=node_number)
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