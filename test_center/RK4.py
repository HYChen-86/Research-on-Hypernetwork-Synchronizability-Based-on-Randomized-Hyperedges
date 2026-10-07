import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importing Axes3D for 3D plotting

def lorenz(t, x, sigma, rho, beta):
    """Lorenz 系统的微分方程"""
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return np.array([dx, dy, dz])

def runge_kutta_lorenz(sigma, rho, beta, x0, dt, num_steps):
    """使用四阶龙格-库塔法离散化 Lorenz 系统"""
    t = np.zeros(num_steps + 1)
    x = np.zeros((num_steps + 1, 3))
    x[0] = x0

    for i in range(num_steps):
        t[i+1] = t[i] + dt

        k1 = lorenz(t[i], x[i], sigma, rho, beta)
        k2 = lorenz(t[i] + dt/2, x[i] + (dt/2) * k1, sigma, rho, beta)
        k3 = lorenz(t[i] + dt/2, x[i] + (dt/2) * k2, sigma, rho, beta)
        k4 = lorenz(t[i] + dt, x[i] + dt * k3, sigma, rho, beta)

        x[i+1] = x[i] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return t, x

# 参数设置
sigma = 10
rho = 28
beta = 8/3
x0 = np.array([0.1, 0, 0])
dt = 0.01
num_steps = 10000

# 使用四阶龙格-库塔法离散化 Lorenz 系统
t, x = runge_kutta_lorenz(sigma, rho, beta, x0, dt, num_steps)

# 绘制轨迹
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x[:, 0], x[:, 1], x[:, 2], lw=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Lorenz System - runge_kutta_lorenz')
plt.show()
