import numpy as np
import matplotlib.pyplot as plt
from BA import BA_non_uniform
from linegraph import weighting_linegraph

# 构建非均匀BA超网络
total_hyper_e_num = 100
# hyper_adjacent_matrix = BA_non_uniform(total_hyper_e_num, 1, 10)  # 构建网络
hyper_adjacent_matrix = np.array(([1,1,1,0,0,0,0],
                                  [0,1,1,1,1,0,0],
                                  [1,0,0,0,1,1,1]))
# 从CSV文件中加载矩阵
filename = f'network_{total_hyper_e_num}.csv'
loaded_matrix = np.loadtxt(filename, delimiter=',')
M = hyper_adjacent_matrix.shape[0]
N = hyper_adjacent_matrix.shape[1]
all_line_graph = weighting_linegraph(hyper_edge_matrix=hyper_adjacent_matrix)  # 转加权线图
a = 0  # 统一混沌系统参数
h = 0
C_1 = 0.0001  # 超边内耦合强度
C_2 = 0.01  # 超边间耦合强度
print("网络构建完成")

# 更新统一混沌系统的函数
def initialization(i, state, t):
    # 节点i前一时刻参数
    x = state[i][t-1][0]
    y = state[i][t-1][1]
    z = state[i][t-1][2]

    # 当前时刻三分量计算
    dxdt = x + ((25 * a + 10) * (y - x))
    dydt = y + ((28 - 35 * a) * x - x * z + (29 * a - 1) * y)
    dzdt = z + (x * y - (8 + a) / 3 * z)

    # 保存当前时刻分量值
    state[i][t][0] = dxdt
    state[i][t][1] = dydt
    state[i][t][2] = dzdt

    return state

# 超边内的同步影响（超邻接矩阵）
def inner(i, adjacent_matrix, state, quantity, t):
    sum = 0
    count = 0
    for k in range(M):
        if adjacent_matrix[k][i] == 1:
            for j in range(N):
                sum += adjacent_matrix[k][j] * (state[j][t-1][quantity] - state[i][t-1][quantity])  # 超边内节点影响之和
                if adjacent_matrix[k][j] != 0 and j != i :
                    count += 1
    return sum

# 超边间的同步影响（超邻接矩阵X加权线图矩阵）
def outer(i, adjacent_matrix, line_graph, state, quantity, t):
    matrix = line_graph @ adjacent_matrix  # 获取相邻超边的节点集合
    sum = 0
    count = 0
    for k in range(M):
        if adjacent_matrix[k][i] == 1:
            for j in range(N):
                sum += matrix[k][j] * (state[j][t-1][quantity] - state[i][t-1][quantity])  # 相邻超边节点影响之和
                if matrix[k][j] != 0 and j != i :
                    count += 1
    return sum

# RK4方法
def RK4(x0, y0, h, n, fs):
    m = len(fs)
    x = np.linspace(x0, x0 + (n - 1) * h, num=n)
    y = np.zeros(shape=(m, n))
    y[:, 0] = y0
    for i in range(n - 1):
        # 计算K1
        K1 = np.zeros(m)
        for index, f in enumerate(fs):
            K1[index] = f(x[i], y[:, i])
        # 计算K2
        K2 = np.zeros(m)
        for index, f in enumerate(fs):
            K2[index] = f(x[i] + h / 2, y[:, i] + h * K1 / 2)
        # 计算K3
        K3 = np.zeros(m)
        for index, f in enumerate(fs):
            K3[index] = f(x[i] + h / 2, y[:, i] + h * K2 / 2)
        # 计算k4
        K4 = np.zeros(m)
        for index, f in enumerate(fs):
            K4[index] = f(x[i] + h, y[:, i] + h * K3)

        y[:, i + 1] = y[:, i] + h * (K1 + 2 * K2 + 2 * K3 + K4) / 6
    return x, y

# 更新统一混沌系统的函数（使用RK4方法）
def update_state(i, C_1, C_2, state, t, dt):
    def dxdt_func(x, y, z):
        return (25 * a + 10) * (y - x) \
            + C_1 * inner(i=i, adjacent_matrix=hyper_adjacent_matrix, state=state, quantity=0, t=t) \
            + C_2 * outer(i=i, adjacent_matrix=hyper_adjacent_matrix, line_graph=all_line_graph, state=state, quantity=0, t=t)

    def dydt_func(x, y, z):
        return (28 - 35 * a) * x - x * z + (29 * a - 1) * y

    def dzdt_func(x, y, z):
        return x * y - (8 + a) / 3 * z \
            + C_2 * outer(i=i, adjacent_matrix=hyper_adjacent_matrix, line_graph=all_line_graph, state=state, quantity=2, t=t)

    # 获取节点i前一时刻的状态
    x, y, z = state[i][t-1]

    # 微分方程组
    fs = [
        lambda _, y: dxdt_func(y[0], y[1], y[2]),
        lambda _, y: dydt_func(y[0], y[1], y[2]),
        lambda _, y: dzdt_func(y[0], y[1], y[2])
    ]

    # 使用RK4方法计算下一时刻的状态
    _, new_state = RK4(0, np.array([x, y, z]), dt, 2, fs)

    # 保存计算结果
    state[i][t] = new_state[:, 1]  # 保存t时刻的新状态

    return state

# # 更新统一混沌系统的函数（使用RK4方法）
# def update_state(i, C_1, C_2, state, t, dt):
#     def dxdt_func(x, y, z):
#         return (25 * a + 10) * (y - x) \
#             + C_1 * inner(i=i, adjacent_matrix=hyper_adjacent_matrix, state=state, quantity=0, t=t) \
#             + C_2 * outer(i=i, adjacent_matrix=hyper_adjacent_matrix, line_graph=all_line_graph, state=state, quantity=0, t=t)

#     def dydt_func(x, y, z):
#         return (28 - 35 * a) * x - x * z + (29 * a - 1) * y

#     def dzdt_func(x, y, z):
#         return x * y - (8 + a) / 3 * z \
#             + C_2 * outer(i=i, adjacent_matrix=hyper_adjacent_matrix, line_graph=all_line_graph, state=state, quantity=2, t=t)
        
#     # 节点i前一时刻参数
#     x = state[i][t-1][0]
#     y = state[i][t-1][1]
#     z = state[i][t-1][2]

   
#     # for _ in range(10):  # 使用 dt = 0.1 进行10次子步骤
#     # RK4步骤
#     k1x = dxdt_func(x, y, z)
#     k1y = dydt_func(x, y, z)
#     k1z = dzdt_func(x, y, z)

#     k2x = dxdt_func(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z)
#     k2y = dydt_func(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z)
#     k2z = dzdt_func(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z)

#     k3x = dxdt_func(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z)
#     k3y = dydt_func(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z)
#     k3z = dzdt_func(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z)

#     k4x = dxdt_func(x + dt * k3x, y + dt * k3y, z + dt * k3z)
#     k4y = dydt_func(x + dt * k3x, y + dt * k3y, z + dt * k3z)
#     k4z = dzdt_func(x + dt * k3x, y + dt * k3y, z + dt * k3z)

#     x_new = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
#     y_new = y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
#     z_new = z + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)

#     #     # 更新节点的状态
#     #     x, y, z = x_new, y_new, z_new

#     # 保存当前时刻分量值
#     state[i][t][0] = k1x
#     state[i][t][1] = k1y
#     state[i][t][2] = k1z

#     return state
    
# 计算标准差误差函数error
def calculate_error(state, t):
    error = 0
    distances = []
    for i in range(N):
        for j in range(N):
            if i != j:  # 避免计算同一个节点之间的差异
                error = 0  # 重置error
                for n in range(3):
                    error += (state[i, t, n] - state[j, t, n]) ** 2
                distance = np.sqrt(error)
                distances.append(distance)
    distances = np.array(distances)  # 将差值列表转换为NumPy数组
    std_dev = np.std(distances, ddof=0)  # 样本标准差使用ddof=0，计算的是整个群体（总体）的标准差
    return std_dev

# 计算均方根误差
def calculate_rmse(state, t):
    total_squared_error = 0
    count = 0
    
    for i in range(N):
        for j in range(N):
            if i != j:  # 避免计算同一个节点之间的差异
                squared_error = 0
                for n in range(3):
                    squared_error += (state[i, t, n] - state[j, t, n]) ** 2
                
                total_squared_error += squared_error
                count += 1
    
    # 计算均方根误差
    mse = total_squared_error / count  # 均方误差 (Mean Squared Error)
    rmse = np.sqrt(mse)  # 均方根误差 (Root Mean Square Error)
    
    return rmse

# 主函数
error_list = []  # 误差列表
num_iterations = 50  # 总时间步
state = np.zeros((N, num_iterations + 1, 3))  # 初始化状态矩阵

# 初始化
t = 0
for i in range(N):
    state[i][0][0] = -16
    state[i][0][1] = -21
    state[i][0][2] = 33 + np.random.uniform(-0.001, 0.001)
result = calculate_rmse(state, t)
error_list.append(result)  # 计算同步误差
print(str(t) + ":" + str(result))
print("初始化完成")

# 前5步，混沌
for t in range(1, 6):
    for i in range(N):
        state = update_state(i=i, C_1=C_1, C_2=C_2, state=state, t=t, dt=0.1)  # 更新每个节点状态
    result = calculate_rmse(state, t)
    error_list.append(result)  # 计算同步误差
    print(str(t) + ":" + str(result))
print("混沌完成")

# 开始同步相互作用
for t in range(6, num_iterations + 1):
    for i in range(N):
        state = update_state(i=i, C_1=C_1, C_2=C_2, state=state, t=t, dt=0.1)
    result = calculate_rmse(state, t)
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
