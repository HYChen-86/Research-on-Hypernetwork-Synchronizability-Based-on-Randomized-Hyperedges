import numpy as np
from numba import cuda

# 获取超网络的 2-section 图的邻接矩阵
def section(hyper_edge_matrix):
    @cuda.jit
    def calculate_2_section(hyper_guanlian, section2):
        total_hyper_e_num = hyper_guanlian.shape[0]
        N = hyper_guanlian.shape[1]
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        bw = cuda.blockDim.x

        start_i = tx + ty * bw
        stride_i = bw * cuda.gridDim.x

        for i in range(start_i, total_hyper_e_num, stride_i):
            for j in range(N):
                for k in range(j + 1, N):  # Start from j+1 to avoid duplicate comparisons
                    if hyper_guanlian[i, j] == 1 and hyper_guanlian[i, k] == 1:
                        section2[j, k] = 1
                        section2[k, j] = 1

    # Assuming hyper_guanlian is already defined
    total_hyper_e_num = len(hyper_edge_matrix)
    N = hyper_edge_matrix.shape[1]
    section = np.zeros((N, N))

    # Configure the CUDA kernel
    threadsperblock = 256
    blockspergrid = (total_hyper_e_num + (threadsperblock - 1)) // threadsperblock

    # Launch the kernel
    calculate_2_section[blockspergrid, threadsperblock](hyper_edge_matrix, section)

    # print('already')
    return section