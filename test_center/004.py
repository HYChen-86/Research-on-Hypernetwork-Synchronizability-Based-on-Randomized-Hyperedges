import numpy as np
import matplotlib.pyplot as plt

# Lorenz 系统的微分方程
def lorenz(x, sigma, rho, beta):
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return np.array([dx, dy, dz])

# 四阶龙格-库塔法离散化 Lorenz 系统
def runge_kutta_lorenz(sigma, rho, beta, x0, dt, num_steps):
    t = np.zeros(num_steps + 1)
    x = np.zeros((num_steps + 1, 3))
    x[0] = x0

    for i in range(num_steps):
        t[i+1] = t[i] + dt

        k1 = lorenz(x[i], sigma, rho, beta)
        k2 = lorenz(x[i] + (dt / 2) * k1, sigma, rho, beta)
        k3 = lorenz(x[i] + (dt / 2) * k2, sigma, rho, beta)
        k4 = lorenz(x[i] + dt * k3, sigma, rho, beta)

        x[i+1] = x[i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, x

# 计算动态方程的梯度
def runge_kutta_step(x, laplacian, sigma, rho, beta, dt):
    def f(x):
        return lorenz(x, sigma, rho, beta)
    
    def H(x):
        return np.array([x[0], 0, x[2]])  # 仅考虑 x 分量的耦合
    
    dxdt = np.zeros_like(x)  # 保存每个节点的变化率
    N = len(x)
    for i in range(N):
        coupling_sum = 0
        for j in range(N):
            coupling_sum += laplacian[i, j] * H(x[j])
        dxdt[i] = f(x[i]) - coupling_sum
    return dxdt

# 使用四阶龙格-库塔法进行同步动力学模拟
def simulate_dynamics(laplacian, initial_state, sigma, rho, beta, dt, num_steps):
    N = len(initial_state)
    state_history = np.zeros((N, num_steps + 1, 3))  # 状态历史记录
    state_history[:, 0, :] = initial_state  # 初始状态
    
    for step in range(num_steps):
        k1 = runge_kutta_step(state_history[:, step, :], laplacian, sigma, rho, beta, dt)
        k2 = runge_kutta_step(state_history[:, step, :] + 0.5 * dt * k1, laplacian, sigma, rho, beta, dt / 2)
        k3 = runge_kutta_step(state_history[:, step, :] + 0.5 * dt * k2, laplacian, sigma, rho, beta, dt / 2)
        k4 = runge_kutta_step(state_history[:, step, :] + dt * k3, laplacian, sigma, rho, beta, dt)
        
        state_history[:, step + 1, :] = state_history[:, step, :] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    error_history = np.zeros(num_steps)  # 保存每个时间步的误差
    for step in range(num_steps):
        error_history[step] = calculate_error(state_history, N, step)
    
    return state_history, error_history

# 计算误差函数
def calculate_error(state, N, t):
    error = 0
    for i in range(N):
        for j in range(N):
            if i != j:  # 避免计算同一个节点之间的差异
                for n in range(3):  # 对 x, y, z 分量进行计算
                    diff = state[i, t, n] - state[j, t, n]
                    error += np.sqrt(diff ** 2)  # 计算每个分量的差的平方根并累加

    return error

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
def supra_laplacian(supra_network, line_graph, C_1, C_2):
    M = supra_network.shape[0]
    N = supra_network.shape[1]

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

# 合并Laplacian
def Matrix_Merge(matrix):
    L = np.zeros((N, N))
    for i in range(M):
            for j in range(M):
                L += matrix[i * N:(i + 1) * N, j * N:(j + 1) * N]
    return L

from BA import BA_non_uniform
from linegraph import weighting_linegraph
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
all_line_graph = weighting_linegraph(hyper_edge_matrix=hyper_adjacent_matrix)  # 转加权线图
a = 0  # 统一混沌系统参数
h = 0
C_1 = 0  # 超边内耦合强度
C_2 = 0  # 超边间耦合强度
print("网络构建完成")

# 参数设置
sigma = 10
rho = 28
beta = 8 / 3
x0 = np.array([0.4, 0.4, 0.2])
dt = 0.001  # 减小时间步长以提高数值稳定性
num_steps = 10000

# 构建示例的拉普拉斯矩阵 (这里使用了一个简单的例子)
# N = 3  # 假设有 3 个节点
# laplacian = np.array([[1, -1, 0],
#                       [-1, 2, -1],
#                       [0, -1, 1]])
N = hyper_adjacent_matrix.shape[1]
matrix = -supra_laplacian(supra_network=Supra_Adjacent(hyper_network=hyper_adjacent_matrix), line_graph=all_line_graph, C_1=C_1, C_2=C_2)
laplacian = Matrix_Merge(matrix)

# 初始化状态并添加微小扰动
initial_state = np.array([x0 for _ in range(N)])
perturbation = 0.01  # 微小扰动
initial_state += np.random.normal(0, perturbation, initial_state.shape)

# 进行模拟
state_history, error_history = simulate_dynamics(laplacian, initial_state, sigma, rho, beta, dt=dt, num_steps=num_steps)

# 绘制轨迹
plt.figure(figsize=(12, 6))

# 绘制 Lorenz 系统轨迹
plt.subplot(1, 2, 1)
for i in range(N):
    plt.plot(state_history[i, :, 0], state_history[i, :, 1], lw=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('State Trajectories')

# 绘制误差曲线
plt.subplot(1, 2, 2)
plt.plot(np.arange(num_steps), error_history)
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.title('Error vs Time Step')

plt.tight_layout()
plt.show()
