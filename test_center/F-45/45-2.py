import numpy as np
import matplotlib.pyplot as plt
from BA import BA_non_uniform_new, BA_non_uniform_old, BA_non_uniform, BA_uniform
# from linegraph import weighting_linegraph, n_linegraph
# from section import section
import os

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

# hyper_adjacent_matrix = np.array([[1,1,0,0,0,0,0,0],
#                                  [1,1,1,1,1,1,1,1]])
# total_hyper_e_num = hyper_adjacent_matrix.shape[0]
# new_node = 6

for total_hyper_e_num in range(2, 21):
    for new_node in range(1, 8):
        second_max_eigenvalue_0 = 0
        Ave_Lambda_2_order_0 = 0
        for time in range(50):
            # hyper_adjacent_matrix = BA_non_uniform_new(total_hyper_e_num=total_hyper_e_num, new_node=new_node, Max_node=10)
            # 调用函数并设置超时
            hyper_adjacent_matrix = run_with_timeout(
                BA_non_uniform_new, 30, total_hyper_e_num=total_hyper_e_num, new_node=new_node, Max_node=10
            )

            M = hyper_adjacent_matrix.shape[0]
            N = hyper_adjacent_matrix.shape[1]
            C_1 = 100  # 超边内耦合强度
            print("网络构建完成")

            Supra_Adjacent_matrix = Supra_Adjacent(hyper_network = hyper_adjacent_matrix)
            L_1 = Laplacian_inner(matrix=Supra_Adjacent_matrix, C=C_1)
            print("Laplacian_Matrix构建完成")

            # 计算特征值
            sorted_eigenvalues, second_max_eigenvalue, eigenvalues = Eigenvalue_ratio(adjacency_matrix = -L_1 / M)
            # print("特征值序列为:", sorted_eigenvalues)
            print("Lambda_2=", second_max_eigenvalue)

            # 计算各超边Fiedler值及平均Fiedler值
            Lambda_2_order, Lambda_order, matrix = Eigenvalue_ratio_mutiorder(adjacency_matrix = Supra_Adjacent_matrix)
            # sorted_eigenvalues = np.sort(Lambda_2_order)
            # print('Lambda_2_order:', Lambda_2_order)
            # print('sorted_Lambda_2_order:',  np.sort(Lambda_2_order))
            Ave_Lambda_2_order = np.average(Lambda_2_order)
            print('average_lambda_2:', Ave_Lambda_2_order)

            second_max_eigenvalue_0 += second_max_eigenvalue
            Ave_Lambda_2_order_0 += Ave_Lambda_2_order

        with open(f'non_uniform_8.txt', 'a') as file:
                file.write(str(total_hyper_e_num) + ',' + str(new_node) + ',' + str(second_max_eigenvalue_0*M/50) + ',' + str(second_max_eigenvalue_0/50) + ',' + str(Ave_Lambda_2_order_0/50) + '\n')  # 在新行追加结果


'''
# 构建非均匀BA超网络
for total_hyper_e_num in range(2, 11):
# total_hyper_e_num = 20
    for node_number in range(1, 8):
    # node_number = 5
        max_node = 10
        for time in range(20):
            hyper_adjacent_matrix = BA_uniform(total_hyper_e_num=total_hyper_e_num, new_node=node_number, k=max_node)  # 构建网络
            # hyper_adjacent_matrix = BA_non_uniform_old(total_hyper_e_num=total_hyper_e_num, old_node=node_number, Max_node=max_node)  # 构建网络


            # # 调用函数并设置超时
            # hyper_adjacent_matrix = run_with_timeout(
            #     BA_non_uniform_new, 30, total_hyper_e_num=total_hyper_e_num, new_node=node_number, Max_node=max_node
            # )

            # 从CSV文件中加载矩阵
            # filename = f'network_{total_hyper_e_num}.csv'
            # hyper_adjacent_matrix = np.loadtxt(filename, delimiter=',')
            M = hyper_adjacent_matrix.shape[0]
            N = hyper_adjacent_matrix.shape[1]
            a = 0  # 统一混沌系统参数
            C_1 = 1  # 超边内耦合强度
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

            # # 计算特征值
            # sorted_eigenvalues, second_max_eigenvalue, eigenvalues = Eigenvalue_ratio(adjacency_matrix = -L_1)
            # print("特征值序列为:", sorted_eigenvalues)
            # print("Lambda_2=", second_max_eigenvalue)

            # 计算平均超边规模
            # Calculating the sum of each row in hyper_adjacent_matrix
            row_sums = np.sum(hyper_adjacent_matrix, axis=1)

            # Calculating the average of these row sums
            average_row_sum = np.mean(row_sums) - node_number


            # with open(f'F-44/{total_hyper_e_num}_{node_number}_{max_node}.txt', 'a') as file:
            #         file.write(str(M) + ',' + str(second_max_eigenvalue) + ',' + str(average_row_sum) + '\n')  # 在新行追加结果

            # 计算特征值
            sorted_eigenvalues_ave, second_max_eigenvalue_ave, eigenvalues_ave = Eigenvalue_ratio(adjacency_matrix = -L_1 / M)
            # print("特征值序列为:", sorted_eigenvalues_ave)
            print("Lambda_2=", second_max_eigenvalue_ave)

            # 计算各超边Fiedler值及平均Fiedler值
            Lambda_2_order, Lambda_order, matrix = Eigenvalue_ratio_mutiorder(adjacency_matrix = Supra_Adjacent_matrix)
            # sorted_eigenvalues = np.sort(Lambda_2_order)
            print('Lambda_2_order:', Lambda_2_order)
            # print('sorted_Lambda_2_order:',  np.sort(Lambda_2_order))
            Ave_Lambda_2_order = np.average(Lambda_2_order)
            print('average_lambda_2:', Ave_Lambda_2_order)
            # print('sum_lambda_2:', np.sum(Lambda_2_order))
            # print('Lambda_order:', Lambda_order)
            count_lambda = 0
            for line in range(M):
                count_lambda += Lambda_2_order[line] * np.count_nonzero(hyper_adjacent_matrix[line, :])
            print('sum_lambda2:', count_lambda)
            target = count_lambda/N/M
            print('average_sum_lambda2:', target)

            # # 与两条超边间的公共节点数有密切关系
            # linegraph_weighting = weighting_linegraph(hyper_edge_matrix = hyper_adjacent_matrix)
            # temp_linegraph = np.zeros(M)
            # count_nn = 0
            # for hyperedge in range(M):
            #     temp_linegraph[hyperedge] = np.sum(linegraph_weighting[hyperedge, :])
            #     count_nn += Lambda_2_order[hyperedge] * np.sum(linegraph_weighting[hyperedge, :])
            # temp = count_nn/N/M
            # print('sum_lambda2(2):', count_nn)
            # print('average_sum_lambda2(2):', temp)

            # os.makedirs("data_center", exist_ok=True)
            # with open(f'data_center/{total_hyper_e_num}_{node_number}_{max_node}.txt', 'a') as file:
            #         file.write(str(second_max_eigenvalue) + ',' + str(temp) + '\n')  # 在新行追加结果


            with open(f'{total_hyper_e_num}_{node_number}_{max_node}.txt', 'a') as file:
                    file.write(str(M) + ',' + str(second_max_eigenvalue_ave) + ',' + str(Ave_Lambda_2_order) + ',' + str(target) + ',' + str(average_row_sum) + '\n')  # 在新行追加结果


'''