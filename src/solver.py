import numpy as np
from numba import cuda


def get_organism_pos(organisms, normalize=False, sim_size=None):
    """
    Return the coordinates of all organisms as a list
    :param sim_size:
    :param organisms: list of organisms
    :param normalize: if 'True', normalize coordinates from -1->1
    :return: list of organism coordinates
    """

    coords = []
    for org in organisms:
        coords.append(org.pos)
    if normalize and sim_size is not None:
        for i in range(len(coords)):
            coords[i][0] /= sim_size[0]/2
            coords[i][1] /= sim_size[1]/2

    return coords


def get_organism_arrays(organisms, GPU=False):

    pos = get_organism_pos(organisms)

    age = []
    clk_counter = []
    clk_lim = []
    clk_sig = []
    health = []
    alive = []
    mov_speed = []
    for org in organisms:
        age.append(org.age)
        clk_counter.append(org.clk_counter)
        clk_lim.append(org.clk_limit)
        clk_sig.append(org.clk_sig)
        health.append(org.health)
        mov_speed.append(org.movement_speed)
        if org.alive:
            alive.append(1)
        else:
            alive.append(0)

    if GPU:
        d_pos = cuda.to_device(pos)
        d_age = cuda.to_device(age)
        d_clk_counter = cuda.to_device(clk_counter)
        d_clk_lim = cuda.to_device(clk_lim)
        d_clk_sig = cuda.to_device(clk_sig)
        d_health = cuda.to_device(health)
        d_alive = cuda.to_device(alive)
        d_mv_speed = cuda.to_device(mov_speed)

        return np.array(pos), np.array(age), np.array(clk_counter), np.array(clk_lim), np.array(clk_sig), \
               np.array(health, dtype='float64'), np.array(alive), np.array(mov_speed),\
               d_pos, d_age, d_clk_counter, d_clk_lim, d_clk_sig, d_health, d_alive, d_mv_speed

    return np.array(pos), np.array(age), np.array(clk_counter), np.array(clk_lim), np.array(clk_sig),\
           np.array(health, dtype='float64'), np.array(alive), np.array(mov_speed)



def any_alive(organisms):
    for org in organisms:
        if org.alive:
            return True
    return False

