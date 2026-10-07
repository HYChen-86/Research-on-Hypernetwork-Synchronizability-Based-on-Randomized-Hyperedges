from BA import BA_non_uniform
import numpy as np
from linegraph import weighting_linegraph, n_linegraph
import matplotlib.pyplot as plt
import copy

# 构建非均匀BA超网络
total_hyper_e_num = 20
# hyper_adjacent_matrix = BA_non_uniform(total_hyper_e_num, 1, 10)  # 构建网络
# hyper_adjacent_matrix = np.array(([1,1,1,0,0,0,0],
#                                   [0,1,1,1,1,0,0],
#                                   [1,0,0,0,1,1,1]))
# 从CSV文件中加载矩阵
filename = f'network_{total_hyper_e_num}.csv'
hyper_adjacent_matrix = np.loadtxt(filename, delimiter=',')

M = hyper_adjacent_matrix.shape[0]
N = hyper_adjacent_matrix.shape[1]
all_line_graph = weighting_linegraph(hyper_edge_matrix=hyper_adjacent_matrix)
# all_line_graph = n_linegraph(hyper_edge_matrix = hyper_adjacent_matrix, target_value=1)  # 转加权线图
a = 0  # 统一混沌系统参数
C_1 = 1  # 超边内耦合强度
C_2 = 1  # 超边间耦合强度
print("网络构建完成")

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

# 构建Supra-Laplacian矩阵(L=A-D)
def supra_laplacian(hyper_network, supra_network, line_graph, C_1, C_2):
    M = hyper_network.shape[0]
    N = hyper_network.shape[1]

    # 创建Supra-Laplacian
    L = np.zeros((N*M, N*M))
    
    # 将线图矩阵扩大为块对角矩阵并放入Laplacian矩阵对应位置
    for i in range(M):
        for j in range(M):
            new_matrix_1 = np.zeros((N, N))
            new_matrix_2 = np.zeros((N, N))
            if i == j:
                # 当 i == j 时，直接取 supra_network 的子矩阵
                new_matrix_1 = supra_network[i, :, :]
            else:
                # 对角线外的情况
                matrix_A = supra_network[i, :, :]
                matrix_B = supra_network[j, :, :]

                if line_graph[i][j] != 0:
                    matrix_C = np.zeros((N, N))
                    matrix_D = np.zeros((N, N))
                    matrix_C = matrix_A @ matrix_B
                    matrix_C[matrix_C != line_graph[i][j]] = 0
                    matrix_D = matrix_B @ matrix_A
                    matrix_D[matrix_D != line_graph[j][i]] = 0
                    new_matrix_2 = matrix_C + matrix_D
                else:
                    new_matrix_2 = np.zeros((N, N))
                # 设置对角线元素为公共节点数
                np.fill_diagonal(new_matrix_2, line_graph[i][j])
            # 将 new_matrix 放置在 Supra-Laplacian 矩阵 L 中正确的位置
            L[i * N:(i + 1) * N, j * N:(j + 1) * N] = C_1 * new_matrix_1 + C_2 * new_matrix_2
            
            # 设置对角线元素为行和的负值
            np.fill_diagonal(L, 0)
            np.fill_diagonal(L, -np.sum(L, axis=1))
                
    return L

A = supra_laplacian(hyper_network=hyper_adjacent_matrix, supra_network=Supra_Adjacent(hyper_network=hyper_adjacent_matrix), line_graph=all_line_graph, C_1=C_1, C_2=C_2)

B = hyper_network=Supra_Adjacent(hyper_network=hyper_adjacent_matrix)
print(A)

# L = np.zeros((N, N))
# for i in range(M):
#         for j in range(M):
#             L += A[i * N:(i + 1) * N, j * N:(j + 1) * N]



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
    second_max_eigenvalue = sorted_eigenvalues[-2]

    # 计算特征值比 R
    R = max_eigenvalue / second_max_eigenvalue

    # print("特征值比 R:", R)
    return sorted_eigenvalues

C = Eigenvalue_ratio(adjacency_matrix=-A)