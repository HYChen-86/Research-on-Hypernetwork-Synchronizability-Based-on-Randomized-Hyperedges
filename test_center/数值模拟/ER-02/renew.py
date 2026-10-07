import numpy as np

# # 保存实验结果
# state_filename = f'20_5_10_old_1500-state_history.npy'
# error_filename = f'20_5_10_old_1500-error_history.npy'

# state_history = np.load(f'20_5_10_old_1500-state_history.npy')
error_history = np.loadtxt(f'20_5_10_old_1500-error_history.txt')


# 保存 state_history 和 error_history 到文件
# np.savetxt(f'20_5_10_old_1500-state_history.txt', state_history)
np.save(f'20_5_10_old_1500-error_history.npy', error_history)