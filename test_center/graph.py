import numpy as np
import matplotlib.pyplot as plt

# 保存实验结果
# state_filename = f'{total_hyper_e_num}_{node_number}_{max_node}_{C_1}-state_history.npy'
# error_filename = f'{total_hyper_e_num}_{node_number}_{max_node}_{C_1}-error_history.npy'
state_filename = f'temp-state_history.npy'
error_filename = f'temp-error_history.npy'

state_history = np.load(state_filename)
error_history = np.load(error_filename)

N = state_history.shape[0]
num_steps = state_history.shape[1] - 1

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