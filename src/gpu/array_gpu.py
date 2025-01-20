import math
import numpy as np
from numba import cuda

"""
Numba/CUDA generic array operation kernels
"""


@cuda.jit
def _flatten(x, start_idx):
    """
    Apply flattening function to [N, 1, M] array
    :param x:
    :param start_idx: specify start node idx
    :return:
    """
    i, j = cuda.grid(2)
    if i < x.shape[0] and j < x.shape[2] and j > start_idx:
        val = x[i, 0, j]
        x[i, 0, j] = math.tanh(val)
        #x[i, 0, j] = val


@cuda.jit
def _matmul(A, W, B, D):
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

    j, m = cuda.grid(2)
    if j < D.shape[2] and m < D.shape[0]:
        tmp = 0.
        for k in range(A.shape[2]):
            tmp += A[m, 0, k] * W[m, k, j]
        D[m, 0, j] = tmp + B[m, 0, j]


# Element-wise multiplication ([A, 1, B] * [A, 1, B])
@cuda.jit
def _self_matrix_elementwise_multiply(A, B):
    i, j = cuda.grid(2)

    if i < A.shape[0] and j < A.shape[2]:
        A[i, 0, j] = A[i, 0, j] * B[i, 0, j]


@cuda.jit
def _set_1d_array_zero(array, ch):
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i = tx + bx * bw  # Compute flattened index inside the array
    if i < array.shape[1]:
        array[ch, i] = 0


@cuda.jit
def _set_2d_array_zero(array, ch):
    x, y = cuda.grid(2)
    if x < array.shape[1] and y < array.shape[2]:
        array[ch, x, y] = 0


@cuda.jit
def _calc_multi_ch_array_2d_grad(array, array_dx, array_dy, ch):
    x, y = cuda.grid(2)
    if x < array.shape[1] and y < array.shape[2]:
        if x == 0:
            array_dx[ch, x, y] = 0
        else:
            array_dx[ch, x, y] = (array[ch, x, y] - array[ch, x - 1, y])
        if y == 1:
            array_dy[ch, x, y] = 0
        else:
            array_dy[ch, x, y] = (array[ch, x, y] - array[ch, x, y - 1])


@cuda.jit
def _copy_2d_array(array_src, array_dst, start_idx, end_idx):
    """
    CUDA kernel
    Copies array 'array_src' elements to 'array_dst'
    elements [0:no_inputs] of array_dst are preserved.
    This assumes inputs are indexed first
    :param array_src:
    :param array_dst:
    :param no_inputs:
    :return:
    """
    x, y = cuda.grid(2)
    if x < array_src.shape[0] and y < array_dst.shape[2]:
        if y >= start_idx and y < end_idx:
            array_dst[x, 0, y] = array_src[x, 0, y]


@cuda.jit
def _zero_and_copy_2d_array(array_src, array_dst, start_idx, end_idx):
    """
    CUDA kernel
    Copies array 'array_src' elements to 'array_dst'
    Zero all elements all array_dst, and copy over elements of array_src from start_idx to end_idx
    This assumes inputs are indexed first
    :param array_src:
    :param array_dst:
    :param no_inputs:
    :return:
    """
    x, y = cuda.grid(2)
    if x < array_src.shape[0] and y < array_dst.shape[2]:
        if y >= start_idx and y < end_idx:
            array_dst[x, 0, y] = array_src[x, 0, y]
        else:
            array_dst[x, 0, y] = 0.


@cuda.jit
def _gated_copy_2d_array(array_src, array_dst, start_idx, end_idx, gate_start_idx, gate_store_threshold):
    """
    CUDA kernel
    Copies array 'array_src' elements to 'array_dst'
    elements [0:no_inputs] of array_dst are preserved. Only copy nodes with gate store signals > threshold, where
    the gate 'store' signals are taken from array_dst at the specified 'gate_start_idx'. The number of gate signals is
    equal to (end_idx - start_idx)
    This assumes inputs are indexed first
    (note: in genetic sim state notation, the array dimensions are typically O[org, 1, nodes]
    :param array_src:
    :param array_dst:
    :param start_idx:
    :param end_idx:
    :param gate_start_idx:
    :param gate_store_threshold:
    :return:
    """
    x, y = cuda.grid(2)
    if x < array_src.shape[0] and y < array_dst.shape[2]:
        if y >= start_idx and y < end_idx:
            gate_no = y - start_idx
            if not array_dst[x, 0, gate_no + gate_start_idx] > gate_store_threshold:
                array_dst[x, 0, y] = array_src[x, 0, y]


@cuda.jit
def _sum_atomic(result, values, sum_idx):
    """
    Sum array of shape [N, M] along dimension M
    Output should be of shape: [N]
    :param result:
    :param array:
    :return:
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    i = (bid * bdim) + tid
    if i < values.shape[1]:
        cuda.atomic.add(result, sum_idx, values[sum_idx, i])


@cuda.jit
def _sum_atomic_channels(result, array, sum_idx, channel_idx):
    """
    Sum array of shape [N, M] along dimension M
    Output should be of shape: [C, M], where C is the number of channels
    :param result: output array of shape [C, M]
    :param array: Array to be summed. Has shape [N, M], where N is the number of elements to be summed
    (i.e. this will sum along dimension 0)
    :return: None
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    i = (bid * bdim) + tid
    if i < array.shape[0]:
        #cuda.atomic.add(result, (channel_idx, sum_idx), array[i, sum_idx]/array.shape[0])
        cuda.atomic.add(result, (channel_idx, sum_idx), array[i, sum_idx]/array.shape[0])



@cuda.jit
def _normalize_array(array, ch):
    """
    Normalize the array along the N dimension.
    :param result: Output array of shape [C, N] for storing the normalized values.
    :param array: Input array of shape [C, N] to be normalized.
    """
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    k = tx + bx * bw  # Compute flattened index inside the array
    if k < 1:
        sum = 0.
        for i in range(array.shape[1]):
            sum += array[ch, i]
        if sum > 0.:
            for i in range(array.shape[1]):
                array[ch, i] /= sum


@cuda.jit
def _average_broadcast(A, row_start, row_end):
    """
    Given an array A of shape [N, _, M], set A[:, _, m] = mean(A[:, _, m]) for all m.
    i.e. broadcast the average of each row to the row elements
    :param A: 2D array
    :param row_start: first row (of N) to start processing
    :param row_end: last row (of N) to process
    :return:
    """
    N, _, M = A.shape  # (organism, 0, node)

    # Get the row index that this thread will process
    row_idx = cuda.grid(1)  # node index

    if row_idx < M and row_idx >= row_start and row_idx < row_end:
        # Calculate the average for the current row
        row_sum = 0.0
        for j in range(N):
            row_sum += A[j, 0, row_idx]
        row_avg = row_sum / N

        # Broadcast the average value to all elements along M (columns)
        for j in range(N):
            A[j, 0, row_idx] = row_avg


def print_gpu_array(gpu_array):
    array_cpu = np.zeros(gpu_array.shape)
    gpu_array.copy_to_host(array_cpu)
    print(array_cpu)


def batch_matmul_offset(A, W, B):
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(A.shape[2] / threadsperblock[0])  # matrix width
    blockspergrid_y = math.ceil(A.shape[0] / threadsperblock[1])  # batch
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    D = np.zeros(A.shape)

    _matmul[blockspergrid, threadsperblock](A, W, B, D)
    return D


def flatten_cuda_array(d_x, start_idx):
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(d_x.shape[0] / threadsperblock[0])  # matrix width
    blockspergrid_y = math.ceil(d_x.shape[2] / threadsperblock[1])  # batch
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _flatten[blockspergrid, threadsperblock](d_x, start_idx)


def test_sum_atomic_normalized():

    no_pos = 12
    channel = 0

    pos = []
    pos.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    pos.append([0.5, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    pos.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1])
    pos = np.array(pos).astype('float')
    d_pos = cuda.to_device(pos)

    # output array
    d_posD = cuda.to_device(np.zeros((1, no_pos)))

    # zero the output array (redundant, but mimicks the actual usage)
    threadsperblock = 16
    blockspergrid = math.ceil(no_pos / threadsperblock)
    _set_1d_array_zero[blockspergrid, threadsperblock](d_posD, channel)

    # Launch the kernel
    pop_count = pos.shape[1]
    threadsperblock = 128
    blockspergrid = math.ceil(pop_count / threadsperblock)
    for ip in range(no_pos):
        _sum_atomic_channels[blockspergrid, threadsperblock](d_posD, d_pos, ip, channel)

    threadsperblock = 16
    blockspergrid = 1
    _normalize_array[blockspergrid, threadsperblock](d_posD, channel)

    # Copy the modified data back from the GPU
    posD = d_posD.copy_to_host()

    gt_posD = pos.sum(0)
    gt_posD = gt_posD/gt_posD.sum()

    # Check the result (print the first row)
    print("Input positions:", pos)
    print("Position density:", posD[channel])
    print("Position density sum:", posD[channel].sum())
    print("(Ground Truth) Position density sum:", gt_posD)


def test_average_broadcast():
    N, M = 10, 20
    A = np.random.rand(N, 1, M).astype(np.float32)

    # Copy the data to the GPU
    d_A = cuda.to_device(A)

    # Configure the kernel grid and block dimensions
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    # Launch the kernel
    _average_broadcast[blocks_per_grid, threads_per_block](d_A, 0, M//2)

    # Copy the modified data back from the GPU
    result = d_A.copy_to_host()

    # Check the result (print the first row)
    print("Original row:", A[:, 0, 1])
    print("Mean of row:", np.mean(A[:, 0, 1]))
    print("Modified row:", result[:, 0, 1])


if __name__ == "__main__":
    #test_average_broadcast()
    test_sum_atomic_normalized()