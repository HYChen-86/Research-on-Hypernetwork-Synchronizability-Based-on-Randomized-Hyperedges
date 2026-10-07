import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 动力学方程 f(x) (以 Lorenz 系统为例)
def f(x, sigma, rho, beta):
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return np.array([dx, dy, dz])

# 耦合函数 H(x) 只在 x 分量上影响，即 diag(1, 0, 0)
def H(x):
    return np.array([x[0], 0, 0])  # 只返回 x[0] 分量，忽略 y 和 z 分量

# 四阶龙格库塔法进行数值积分
def runge_kutta_step(x, laplacian, sigma, rho, beta, C1, dt):
    N = x.shape[0]  # 节点数
    dxdt = np.zeros_like(x)  # 保存每个节点的变化率

    # 计算每个节点的 dx/dt
    for i in range(N):
        coupling_sum = np.zeros(3)  # 存储耦合项
        for j in range(N):
            coupling_sum += laplacian[i, j] * H(x[j])  # 累加耦合项（只在 x 分量上影响）
        dxdt[i] = f(x[i], sigma, rho, beta) - C1 * coupling_sum  # 动力学方程

    return dxdt

# 主函数
def simulate_dynamics(laplacian, initial_state, sigma, rho, beta, C1, dt, num_steps):
    N = initial_state.shape[0]  # 节点数
    state_history = np.zeros((num_steps + 1, N, 3))  # 保存每个时间步的状态
    state_history[0] = initial_state

    # 动力学演化
    for step in range(num_steps):
        x = state_history[step]
        
        # 计算 k1, k2, k3, k4
        k1 = runge_kutta_step(x, laplacian, sigma, rho, beta, C1, dt)
        k2 = runge_kutta_step(x + 0.5 * dt * k1, laplacian, sigma, rho, beta, C1, dt)
        k3 = runge_kutta_step(x + 0.5 * dt * k2, laplacian, sigma, rho, beta, C1, dt)
        k4 = runge_kutta_step(x + dt * k3, laplacian, sigma, rho, beta, C1, dt)

        # 更新状态
        new_state = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # 添加范围限制，防止数值溢出
        new_state = np.clip(new_state, -100, 100)  # 根据系统的特性进行调整

        state_history[step + 1] = new_state

    return state_history

# 参数设置
sigma = 10
rho = 28
beta = 8/3
C1 = 10  # 减小 C1 以避免数值不稳定
dt = 0.001  # 减小时间步长
num_steps = 10000
N = 3  # 假设有3个节点

# 初始化状态（每个节点初始位置）
initial_state = np.random.rand(N, 3) * 2 - 1  # 初始条件范围减小

# 定义简单的Laplacian矩阵 (例如3个节点的Laplacian)
laplacian = np.array([[2, -1, -1],
                      [-1, 2, -1],
                      [-1, -1, 2]])

# 模拟动力学
state_history = simulate_dynamics(laplacian, initial_state, sigma, rho, beta, C1, dt, num_steps)

# 绘制节点的状态演化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(N):
    ax.plot(state_history[:, i, 0], state_history[:, i, 1], state_history[:, i, 2], lw=0.5, label=f'Node {i+1}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Dynamics of Nodes with Coupling on x Component')
plt.legend()
plt.show()
