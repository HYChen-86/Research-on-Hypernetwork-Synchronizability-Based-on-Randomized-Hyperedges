import numpy as np
import matplotlib.pyplot as plt

def lorenz(X, sigma=10, beta=8/3, rho=28):
    x, y, z = X
    dxdt = -sigma * (x - y)
    dydt = rho * x - y - x * z
    dzdt = -beta * z + x * y
    return np.array([dxdt, dydt, dzdt])

def sync_dynamics(X, N, a, C1, C2, sigma=10, beta=8/3, rho=28):
    H = np.diag([1, 0, 0])
    Gamma = np.diag([1, 0, 1])
    
    X = X.reshape((N, 3))
    dXdt = np.zeros_like(X)
    
    for i in range(N):
        coupling_H = np.sum([a[i, j] * (H @ (X[i] - X[j])) for j in range(N)], axis=0)
        coupling_Gamma = np.sum([a[i, j] * (Gamma @ (X[i] - X[j])) for j in range(N)], axis=0)
        dXdt[i] = lorenz(X[i], sigma, beta, rho) + C1 * coupling_H + C2 * coupling_Gamma
        
    return dXdt.flatten()

def runge_kutta_4_step(f, X, t, dt, *args):
    k1 = f(X, *args)
    k2 = f(X + 0.5 * dt * k1, *args)
    k3 = f(X + 0.5 * dt * k2, *args)
    k4 = f(X + dt * k3, *args)
    return X + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# 初始条件和参数
N = 10  # 系统数量
np.random.seed(0)
X0 = 20 * np.random.rand(N, 3)  # 随机初始条件
X0 = X0.flatten()

a = np.ones((N, N)) - np.eye(N)  # 全连接的邻接矩阵
C1, C2 = 0.1, 0.1
dt = 0.1
total_steps = 500

# 时间数组
t = np.arange(0, total_steps * dt, dt)

# 初始化结果数组
X = np.zeros((total_steps, N, 3))
X[0] = X0.reshape((N, 3))

# 使用四阶Runge-Kutta法求解
for i in range(1, total_steps):
    X[i] = runge_kutta_4_step(sync_dynamics, X[i-1].flatten(), t[i-1], dt, N, a, C1, C2).reshape((N, 3))

def global_sync_error(X):
    mean_state = np.mean(X, axis=0)
    error = np.mean(np.linalg.norm(X - mean_state, axis=1))
    return error

sync_errors = np.array([global_sync_error(X_t) for X_t in X])

plt.plot(t, sync_errors)
plt.xlabel('Time')
plt.ylabel('Global Synchronization Error')
plt.title('Global Synchronization Error Over Time')
plt.show()
