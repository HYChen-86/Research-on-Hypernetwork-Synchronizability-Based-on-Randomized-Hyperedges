from BA import BA_non_uniform
import numpy as np
from linegraph import weighting_linegraph
import matplotlib.pyplot as plt

# 构建非均匀BA超网络
total_hyper_e_num = 100
hyper_adjacent_matrix = BA_non_uniform(total_hyper_e_num, 1, 10)  # 构建网络
M = hyper_adjacent_matrix.shape[0]
N = hyper_adjacent_matrix.shape[1]
all_line_graph = weighting_linegraph(hyper_edge_matrix = hyper_adjacent_matrix)  # 转加权线图
a = 0  # 统一混沌系统参数
C_1 = 4  # 超边内耦合强度
C_2 = 2  # 超边间耦合强度
print("网络构建完成")

# 构建三维邻接矩阵Supra-Adjacent
def Supra_Adjacent(hyper_network):
    M = hyper_network.shape[0]
    N = hyper_network.shape[1]
    structural_matrix = np.zeros((M, N, N))
    for k in range(M):
        for j in range(N):
            for i in range(j, N):
                if hyper_network[k][j] == hyper_network[k][i] == 1:
                    structural_matrix[k][i][j] = 1
                    structural_matrix[k][j][i] = 1

    return structural_matrix

# 构建Supra-Laplacian矩阵(L=A-D)
def supra_laplacian(hyper_network, line_graph, C_1, C_2):
    M = hyper_network.shape[0]
    N = hyper_network.shape[1]

    # 创建Supra-Laplacian
    L = np.zeros((N*M, N*M))

    # 处理两个矩阵的主对角线
    for a in range(M):
        # 将线图矩阵的对角线值置为每行之和的相反数
        row_sum = np.sum(line_graph[a])
        line_graph[a, a] = -row_sum
        # 将Supra-Adjacent的每一层对角线值置为每行之和的相反数
        for b in range(N):
            row_sum = np.sum(hyper_network[a][b])
            hyper_network[a, b, b] = -row_sum

    # 将线图矩阵扩大为块对角矩阵并放入Laplacian矩阵对应位置
    for i in range(M):
        for j in range(M):
            # 创建一个新的二维数组，大小为(N, N)
            new_matrix = np.zeros((N, N))
            for k in range(N):
                new_matrix[k, k] = line_graph[i][j]
            # 将new_matrix放入Supra-Laplacian矩阵的对应位置
            L[i*N:(i+1)*N, j*N:(j+1)*N] += C_2 * new_matrix

    # 将Supra-Adjacent的每一层矩阵放入Laplacian矩阵对应位置
    for c in range(M):
        for d in range(N):
            L[c*d:(c+1)*d, c*d:(c+1)*d] += C_1 * hyper_network[c]

    return L

# 更新统一混沌系统的函数（使用RK4方法）
def update_state(i, C_1, C_2, state, t, dt):
    def dxdt_func(x, y, z):
        return (25 * a + 10) * (y - x) \
               + C_1 * inner(i=i, adjacent_matrix=hyper_adjacent_matrix, state=state, quantity=1, t=t) \
               + C_2 * outer(i=i, adjacent_matrix=hyper_adjacent_matrix, line_graph=all_line_graph, state=state, quantity=1, t=t)

    def dydt_func(x, y, z):
        return (28 - 35 * a) * x - x * z + (29 * a - 1) * y

    def dzdt_func(x, y, z):
        return x * y - (8 + a) / 3 * z \
               + C_2 * outer(i=i, adjacent_matrix=hyper_adjacent_matrix, line_graph=all_line_graph, state=state, quantity=3, t=t)
    
    # 节点i前一时刻参数
    x = state[i][t-1][1]
    y = state[i][t-1][2]
    z = state[i][t-1][3]

    # RK4步骤
    k1x = dxdt_func(x, y, z)
    k1y = dydt_func(x, y, z)
    k1z = dzdt_func(x, y, z)

    k2x = dxdt_func(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z)
    k2y = dydt_func(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z)
    k2z = dzdt_func(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z)

    k3x = dxdt_func(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z)
    k3y = dydt_func(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z)
    k3z = dzdt_func(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z)

    k4x = dxdt_func(x + dt * k3x, y + dt * k3y, z + dt * k3z)
    k4y = dydt_func(x + dt * k3x, y + dt * k3y, z + dt * k3z)
    k4z = dzdt_func(x + dt * k3x, y + dt * k3y, z + dt * k3z)

    x_new = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    y_new = y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
    z_new = z + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)

    # 保存当前时刻分量值
    state[i][t][1] = x_new
    state[i][t][2] = y_new
    state[i][t][3] = z_new

    return state

# 超边内的同步影响（超邻接矩阵）
def inner(i, adjacent_matrix, state, quantity, t):
    matrix = adjacent_matrix
    sum = 0
    for k in range(M):
        if matrix[k][i] == 1:
            for j in range(N):
                if j != i:
                    sum += matrix[k][j] * (state[j][t-1][quantity] - state[i][t-1][quantity])  # 超边内节点影响之和
    return sum

# 超边间的同步影响（超邻接矩阵X加权线图矩阵）
def outer(i, adjacent_matrix, line_graph, state, quantity, t):
    matrix = np.dot(line_graph, adjacent_matrix)  # 获取相邻超边的节点集合
    # matrix = line_graph * adjacent_matrix  # 获取相邻超边的节点集合
    sum = 0
    for k in range(M):
        if matrix[k][i] == 1:
            for j in range(N):
                if j != i:
                    sum += matrix[k][j] * (state[j][t-1][quantity] - state[i][t-1][quantity])  # 相邻超边节点影响之和
    return sum

# 误差函数error
def calculate_error(state, t):
    error = 0
    for i in range(N):
        for j in range(N):
            for n in range(1, 4):
                error += np.sqrt((state[i, t, n] - state[j, t, n]) ** 2)
    return error


# 主函数
error_list = []  # 误差列表
num_iterations = 500  # 总时间步
state = np.zeros((N, num_iterations, 4)) * 1e20  # 初始化状态矩阵

# 初始化
t = 0
for i in range(N):
    state[i][0][1] = np.random.randint(40)
    state[i][0][2] = np.random.randint(40)
    state[i][0][3] = np.random.randint(20)
result = calculate_error(state, t)
error_list.append(result)  # 计算同步误差
print(str(t) + ":" + str(result))
print("初始化完成")

# # 前5步，混沌
# for t in range(1, 6):
#     for i in range(N):
#         state = update_state(i=i, C_1=0, C_2=0, state=state, t=t)  # 更新每个节点状态
#     result = calculate_error(state, t)
#     error_list.append(result)  # 计算同步误差
#     print(str(t) + ":" + str(result))
# print("混沌完成")

# 开始同步相互作用
for t in range(1, num_iterations+1):
    for i in range(N):
        state = update_state(i=i, C_1=C_1, C_2=C_2, state=state, t=t, dt=0.1)  # 更新每个节点状态
    result = calculate_error(state, t)
    error_list.append(result)  # 计算同步误差
    print(str(t) + ":" + str(result))
print("同步模拟完成")

# 绘制误差曲线并保存（写入txt）
with open('error_data.txt', 'w') as file:
    for error in error_list:
        file.write(f"{error}\n")

# 绘制误差曲线
plt.plot(error_list, label='Error')
plt.xlabel('Index')
plt.ylabel('Error')
plt.title('Error Curve')
plt.legend()
plt.grid(True)
plt.savefig('error_curve.png')  # 保存为PNG图片
plt.show()  # 显示图表

# 计算Lambda2（不同耦合强度下）


# 绘制Lambda曲线并保存（写入txt）


