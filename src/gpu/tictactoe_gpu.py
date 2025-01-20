import numpy as np
from numba import cuda, config
import math

@cuda.jit
def _Update_Game_State(O, decision_offset, decision_count):
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    if i_org < O.shape[0]:
        decision = O[i_org, 0, decision_offset:(decision_offset + decision_count)]


@cuda.jit
def _count_predictions_above_threshold(O, threshold, decision_offset, no_decisions, decisions):
    i, j = cuda.grid(2)

    if i < O.shape[0] and j < no_decisions:
        if O[i, 0, j+decision_offset] > threshold:
            cuda.atomic.add(decisions, j, 1)

def test_count_predictions():
    # Example usage
    N = 5  # Number of models
    M = 10  # Number of predictions per model
    threshold = 0.5
    A = np.random.rand(N, 1, M).astype(np.float32)  # Example prediction array

    print(A)

    # Copy data to GPU
    d_A = cuda.to_device(A)
    d_result = cuda.to_device(np.zeros(M, dtype=np.int32))

    # Define grid and block dimensions
    threadsperblock = (16, 16)
    blockspergrid_x = (A.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (A.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Launch kernel
    _count_predictions_above_threshold[blockspergrid, threadsperblock](d_A, threshold, 0, d_result)

    # Copy result back to host
    result = d_result.copy_to_host()
    print(result)