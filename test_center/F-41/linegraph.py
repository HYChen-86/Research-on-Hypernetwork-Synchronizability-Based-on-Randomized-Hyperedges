import numpy as np
from numba import cuda

# 计算超边的度（超图的n线图矩阵）
# target_value = 2  # 设置n线图的n 

# class Linegraph:

def n_linegraph(hyper_edge_matrix, target_value):
    @cuda.jit
    def calculate_n_linegraph(hyper_edge_matrix, hyperedge_matrix, target_value):
        total_hyper_e_num = hyperedge_matrix.shape[0]
        N = hyper_edge_matrix.shape[1]
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        bw = cuda.blockDim.x

        start = tx + ty * bw
        stride = bw * cuda.gridDim.x

        for index in range(start, total_hyper_e_num, stride):
            a = index
            for b in range(a + 1, total_hyper_e_num):
                flag_edge = 0
                for c in range(N):
                    if hyper_edge_matrix[b, c] != 0 and hyper_edge_matrix[a, c] != 0 and hyper_edge_matrix[a, c] == hyper_edge_matrix[b, c]:
                        flag_edge += 1
                    if flag_edge == target_value:
                        break
                if flag_edge == target_value:
                    hyperedge_matrix[a, b] = 1
                    hyperedge_matrix[b, a] = 1

    # Assuming hyper_edge_matrix is already defined
    total_hyper_e_num = len(hyper_edge_matrix)
    hyperedge_matrix = np.zeros((total_hyper_e_num, total_hyper_e_num))

    # Configure the CUDA kernel
    threadsperblock = 256
    blockspergrid = (total_hyper_e_num + (threadsperblock - 1)) // threadsperblock

    # Launch the kernel
    calculate_n_linegraph[blockspergrid, threadsperblock](hyper_edge_matrix, hyperedge_matrix, target_value)

    # print(str(target_value) + "-line graph:\n", hyperedge_matrix)  
    return hyperedge_matrix


def weighting_linegraph(hyper_edge_matrix):
    @cuda.jit
    def calculate_weighting_linegraph(hyper_edge_matrix, hyperedge_matrix):
        total_hyper_e_num = hyperedge_matrix.shape[0]
        N = hyper_edge_matrix.shape[1]
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        bw = cuda.blockDim.x

        start = tx + ty * bw
        stride = bw * cuda.gridDim.x

        for index in range(start, total_hyper_e_num, stride):
            a = index
            for b in range(a + 1, total_hyper_e_num):
                flag_edge = 0
                for c in range(N):
                    if hyper_edge_matrix[b, c] != 0 and hyper_edge_matrix[a, c] != 0 and hyper_edge_matrix[a, c] == hyper_edge_matrix[b, c]:
                        flag_edge += 1
                
                hyperedge_matrix[a, b] = flag_edge
                hyperedge_matrix[b, a] = flag_edge

    # Assuming hyper_edge_matrix is already defined
    total_hyper_e_num = len(hyper_edge_matrix)
    hyperedge_matrix = np.zeros((total_hyper_e_num, total_hyper_e_num))

    # Configure the CUDA kernel
    threadsperblock = 256
    blockspergrid = (total_hyper_e_num + (threadsperblock - 1)) // threadsperblock

    # Launch the kernel
    calculate_weighting_linegraph[blockspergrid, threadsperblock](hyper_edge_matrix, hyperedge_matrix)

    # print("all-line graph:\n", hyperedge_matrix)  
    return hyperedge_matrix

def weighting_n_linegraph(hyper_edge_matrix, target_value):
    @cuda.jit
    def calculate_weighting_n_linegraph(hyper_edge_matrix, hyperedge_matrix, target_value):
        total_hyper_e_num = hyperedge_matrix.shape[0]
        N = hyper_edge_matrix.shape[1]
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        bw = cuda.blockDim.x

        start = tx + ty * bw
        stride = bw * cuda.gridDim.x

        for index in range(start, total_hyper_e_num, stride):
            a = index
            for b in range(a + 1, total_hyper_e_num):
                flag_edge = 0
                for c in range(N):
                    if hyper_edge_matrix[b, c] != 0 and hyper_edge_matrix[a, c] != 0 and hyper_edge_matrix[a, c] == hyper_edge_matrix[b, c]:
                        flag_edge += 1
                if flag_edge >= target_value:
                    hyperedge_matrix[a, b] = flag_edge
                    hyperedge_matrix[b, a] = flag_edge

    # Assuming hyper_edge_matrix is already defined
    total_hyper_e_num = len(hyper_edge_matrix)
    hyperedge_matrix = np.zeros((total_hyper_e_num, total_hyper_e_num))

    # Configure the CUDA kernel
    threadsperblock = 256
    blockspergrid = (total_hyper_e_num + (threadsperblock - 1)) // threadsperblock

    # Launch the kernel
    calculate_weighting_n_linegraph[blockspergrid, threadsperblock](hyper_edge_matrix, hyperedge_matrix, target_value)

    # print("weighting_" + str(target_value) + "-line graph:\n", hyperedge_matrix)  
    return hyperedge_matrix

