import math
import numpy as np
from numba import jit


"""
Numba/CUDA generic array operation kernels
"""

@jit(nopython=True)
def flatten(x, start_idx):
    """
    Apply flattening function to [N, 1, M] array
    :param x:
    :return:
    """
    for i in range(x.shape[0]):
        for j in range(start_idx, x.shape[2]):
            val = x[i, 0, j]
            x[i, 0, j] = math.tanh(val)


@jit(nopython=True)
def matmul(A, W, B, D):
    """
      Perform square matrix multiplication of D = A * W + B
      Each thread finds the closest match for a single vertex 'pos' with all other vertices
      tx = cuda.threadIdx.x  # Thread id in a 1D block
      bx = cuda.blockIdx.x  # Block id in a 1D grid
      bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
      ty = cuda.threadIdx.y  # Thread id in a 1D block
      by = cuda.blockIdx.y  # Block id in a 1D grid
      by = cuda.blockDim.y  # Block width, i.e. number of threads per block
      pos_x = tx + bx * bw  # Compute flattened index inside the array
      pos_y = ty + by * by  # Compute flattened index inside the array
      """

    for j in range(D.shape[2]):
        for m in range(D.shape[0]):
            tmp = 0.
            for k in range(A.shape[2]):
                tmp += A[m, 0, k] * W[m, k, j]
            D[m, 0, j] = tmp + B[m, 0, j]


@jit(nopython=True)
def sum_atomic_channels_normalized(array, sum_idx, channel_idx):
    """
    Sum array of shape [N, M] along dimension M
    Output should be of shape: [C, M], where C is the number of channels
    :param result: output array of shape [C, M]
    :param array: Array to be summed. Has shape [N, M], where N is the number of elements to be summed
    (i.e. this will sum along dimension 0)
    :return: None
    """
    temp = 0
    for i in range(array.shape[0]):
        temp += array[i, sum_idx]/array.shape[0]
    return temp
