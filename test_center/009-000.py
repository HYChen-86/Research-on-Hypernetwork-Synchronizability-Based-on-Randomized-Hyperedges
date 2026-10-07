import numpy as np
import matplotlib.pyplot as plt
from BA import BA_non_uniform, BA_non_uniform_old, BA_non_uniform_new
from linegraph import weighting_linegraph, n_linegraph
from section import section
import threading
import time

# Lorenz 系统的微分方程
def lorenz(x, a):
    dx = (25 * a + 10) * (x[1] - x[0])
    dy = (28 - 35 * a) * x[0] - x[0] * x[2] + (29 * a - 1) * x[1]
    dz = x[0] * x[1] - (8 + a) / 3 * x[2]
    return np.array([dx, dy, dz])

# 四阶龙格-库塔法离散化 Lorenz 系统
def runge_kutta_lorenz(a, x0, dt, num_steps):
    t = np.zeros(num_steps + 1)
    x = np.zeros((num_steps + 1, 3))
    x[0] = x0

    for i in range(num_steps):
        t[i+1] = t[i] + dt

        k1 = lorenz(x[i], a)
        k2 = lorenz(x[i] + (dt / 2) * k1, a)
        k3 = lorenz(x[i] + (dt / 2) * k2, a)
        k4 = lorenz(x[i] + dt * k3, a)

        x[i+1] = x[i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, x

# 计算动态方程的梯度
def runge_kutta_step(x, laplacian_1, a, dt):
    def f(x):
        return lorenz(x, a)
    
    def H(x):
        return np.array([x[0], 0, 0])  # 耦合函数H(x), 仅考虑x分量的耦合
    
    dxdt = np.zeros_like(x)  # 保存每个节点的变化率
    N = len(x)
    for i in range(N):
        coupling_sum_1 = 0
        for j in range(N):
            coupling_sum_1 += (laplacian_1[i, j] * H(x[j]))
        dxdt[i] = f(x[i]) + coupling_sum_1

    return dxdt

# 使用四阶龙格-库塔法进行同步动力学模拟
def simulate_dynamics(laplacian_1, initial_state, a, dt, num_steps):
    N = len(initial_state)
    state_history = np.zeros((N, num_steps + 1, 3))  # 状态历史记录
    state_history[:, 0, :] = initial_state  # 初始状态
    
    for step in range(num_steps):
        k1 = runge_kutta_step(state_history[:, step, :], laplacian_1, a, dt)
        k2 = runge_kutta_step(state_history[:, step, :] + 0.5 * dt * k1, laplacian_1, a, dt / 2)
        k3 = runge_kutta_step(state_history[:, step, :] + 0.5 * dt * k2, laplacian_1, a, dt / 2)
        k4 = runge_kutta_step(state_history[:, step, :] + dt * k3, laplacian_1, a, dt)
        
        state_history[:, step + 1, :] = state_history[:, step, :] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    error_history = np.zeros(num_steps)  # 保存每个时间步的误差
    for step in range(num_steps):
        error_history[step] = calculate_error(state_history, N, step)
    
    return state_history, error_history

# 计算误差函数
def calculate_error(state, N, t):
    error = 0
    distances = []
    for i in range(N):
        for j in range(N):
            if i != j:  # 避免计算同一个节点之间的差异
                error = 0  # 重置error
                for n in range(3):  # 对 x, y, z 分量进行计算
                    diff = state[i, t, n] - state[j, t, n]
                    error += np.sqrt(diff ** 2)  # 计算每个分量的差的平方根并累加
                distance = np.sqrt(error)
                distances.append(distance)
    distances = np.array(distances)  # 将差值列表转换为NumPy数组
    std_dev = np.std(distances, ddof=0)  # 样本标准差使用ddof=0，计算的是整个群体（总体）的标准差

    return std_dev

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

# 构建非均匀BA超网络
total_hyper_e_num = 20
node_number = 2
max_node = 10
# hyper_adjacent_matrix = BA_non_uniform(total_hyper_e_num, 5, 10)  # 构建网络

# 调用函数并设置超时
hyper_adjacent_matrix = run_with_timeout(
    BA_non_uniform_old, 30, total_hyper_e_num=total_hyper_e_num, old_node=node_number, Max_node=max_node
)

# 从CSV文件中加载矩阵
# filename = f'network_{total_hyper_e_num}.csv'
# hyper_adjacent_matrix = np.loadtxt(filename, delimiter=',')
M = hyper_adjacent_matrix.shape[0]
N = hyper_adjacent_matrix.shape[1]
a = 0  # 统一混沌系统参数
C_1 = 600  # 超边内耦合强度
print("网络构建完成")
# print(hyper_adjacent_matrix)

# 参数设置
x0 = np.array([40.0, 40.0, 20.0])
dt = 0.001  # 减小时间步长以提高数值稳定性
num_steps = 50000

# 构建示例的拉普拉斯矩阵
N = hyper_adjacent_matrix.shape[1]
Supra_Adjacent_matrix = Supra_Adjacent(hyper_network = hyper_adjacent_matrix)
L_1 = Laplacian_inner(matrix=Supra_Adjacent_matrix, C=C_1)
print("Laplacian_Matrix构建完成")

# 初始化状态并添加微小扰动
initial_state = np.array([x0 for _ in range(N)])
perturbation = 10.0  # 微小扰动
initial_state += np.random.normal(0, perturbation, initial_state.shape)

# 进行模拟
state_history, error_history = simulate_dynamics(laplacian_1=L_1, initial_state=initial_state, a=a, dt=dt, num_steps=num_steps)

# 保存实验结果
state_filename = f'{total_hyper_e_num}_{node_number}_{max_node}_old_{C_1}-state_history.npy'
error_filename = f'{total_hyper_e_num}_{node_number}_{max_node}_old_{C_1}-error_history.npy'
# state_filename = f'temp-state_history.npy'
# error_filename = f'temp-error_history.npy'

# 保存 state_history 和 error_history 到文件
np.save(state_filename, state_history)
np.save(error_filename, error_history)

state_history = np.load(state_filename)
error_history = np.load(error_filename)

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
