import math
import numpy as np
from gpu_src.Array_GPU import _Set_2DArray_Zero, _Calc_multi_ch_Array_2D_Grad
from gpu_src.GeneticSim_GPU import _Calc_Pheromone_2D


def Calc_Pheromone(environment, population, min_radius, max_radius, sat_level=None, pher_channel=0):
    threadsperblock = (8, 8)
    blockspergrid_x = math.ceil(environment.sim_size[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(environment.sim_size[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    if sat_level is None:
        sat_level = 1E8
    _Set_2DArray_Zero[blockspergrid, threadsperblock](environment.d_pher, pher_channel)  # Reset pheromone array to zeros
    _Calc_Pheromone_2D[blockspergrid, threadsperblock](environment.d_pher, population.d_pos, population.d_pher_rel,
                                                    environment.sim_size[0], environment.sim_size[1], max_radius,
                                                    min_radius, sat_level, pher_channel)
    _Calc_multi_ch_Array_2D_Grad[blockspergrid, threadsperblock](environment.d_pher,
                                                         environment.d_pher_dx, environment.d_pher_dy,
                                                         pher_channel)


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt

