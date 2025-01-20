import math
from src.gpu.genetic_sim_gpu import _apply_position_mask
from src.cpu.genetic_sim_cpu import apply_position_mask

def apply_mask(pos, pos_mask=None, USE_GPU=True):
    if pos_mask is None:
        return
    if USE_GPU:
        if pos.shape[0] < 2000:
            threadsperblock = 8
        elif pos.shape[0] < 4000:
            threadsperblock = 16
        else:
            threadsperblock = 32
        blockspergrid = math.ceil(pos.shape[0] / threadsperblock)
        _apply_position_mask[blockspergrid, threadsperblock](pos, pos_mask)
    else:
        apply_position_mask(pos, pos_mask)

