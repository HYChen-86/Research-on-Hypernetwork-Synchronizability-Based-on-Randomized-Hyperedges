from BA import BA_non_uniform
import numpy as np
from linegraph import weighting_linegraph, n_linegraph
import matplotlib.pyplot as plt
import copy

# 构建非均匀BA超网络
total_hyper_e_num = 20
hyper_adjacent_matrix = BA_non_uniform(total_hyper_e_num, 1, 10)  # 构建网络

filename = f'network_{total_hyper_e_num}.csv'
# 将矩阵保存为CSV文件
np.savetxt(filename, hyper_adjacent_matrix, delimiter=',')

# 从CSV文件中加载矩阵
loaded_matrix = np.loadtxt(filename, delimiter=',')
