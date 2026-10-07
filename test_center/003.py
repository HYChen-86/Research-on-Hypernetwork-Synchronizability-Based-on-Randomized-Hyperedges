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
def runge_kutta_step(x, laplacian, sigma, rho, beta, C1, dt):
    def f(x):
        return lorenz(x, sigma, rho, beta)
    
    def H(x):
        return np.array([x[0], x[1], x[2]])  # 仅考虑 x 分量的耦合
    
    dxdt = np.zeros_like(x)  # 保存每个节点的变化率
    N = len(x)
    for i in range(N):
        coupling_sum = 0
        for j in range(N):
            coupling_sum += laplacian[i, j] * H(x[j])
        dxdt[i] = f(x[i]) - C1 * coupling_sum
    return dxdt

# 使用四阶龙格-库塔法进行同步动力学模拟
def simulate_dynamics(laplacian, initial_state, sigma, rho, beta, C1, dt, num_steps):
    N = len(initial_state)
    state_history = np.zeros((N, num_steps + 1, 3))  # 状态历史记录
    state_history[:, 0, :] = initial_state  # 初始状态
    
    for step in range(num_steps):
        k1 = runge_kutta_step(state_history[:, step, :], laplacian, sigma, rho, beta, C1, dt)
        k2 = runge_kutta_step(state_history[:, step, :] + 0.5 * dt * k1, laplacian, sigma, rho, beta, C1, dt / 2)
        k3 = runge_kutta_step(state_history[:, step, :] + 0.5 * dt * k2, laplacian, sigma, rho, beta, C1, dt / 2)
        k4 = runge_kutta_step(state_history[:, step, :] + dt * k3, laplacian, sigma, rho, beta, C1, dt)
        
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

# 参数设置
sigma = 10
rho = 28
beta = 8 / 3
x0 = np.array([40.0, 40.0, 20.0])
dt = 0.001  # 减小时间步长以提高数值稳定性
num_steps = 10000

# 构建示例的拉普拉斯矩阵 (这里使用了一个简单的例子)
N = 3  # 假设有 3 个节点
laplacian = np.array([[1, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 1]])

# 初始化状态并添加微小扰动
initial_state = np.array([x0 for _ in range(N)])
perturbation = 10  # 微小扰动
initial_state += np.random.normal(0, perturbation, initial_state.shape)

# 进行模拟
state_history, error_history = simulate_dynamics(laplacian, initial_state, sigma, rho, beta, C1=10, dt=dt, num_steps=num_steps)

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
