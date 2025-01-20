import math
from src.gpu.genetic_sim_gpu import _position_change
from src.cpu.genetic_sim_cpu import position_change
from src.species import Population


def update_position_change(population: Population, USE_GPU=True):
    if USE_GPU:
        if population.pop_count < 2000:
            threadsperblock = 8
        elif population.pop_count < 4000:
            threadsperblock = 16
        else:
            threadsperblock = 32
        blockspergrid = math.ceil(population.d_pos.shape[0] / threadsperblock)
        _position_change[blockspergrid, threadsperblock](population.d_pos, population.d_pos_last, population.d_pos_change)
    else:
        position_change(population.pos, population.pos_last, population.pos_change)

