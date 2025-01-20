import numpy as np
import math
from src.gpu.genetic_sim_gpu import _decide_trader, _decide_grazer, _decide_predator, _decide_gamer
from src.cpu.genetic_sim_cpu import decide_trader


def calc_decide_numerical(population, USE_GPU=True):

    # Find output node indexes
    out_start_idx = population.out_offset
    out_pos_idx = out_start_idx

    out_clk_lim_plus = population.template.out_idx["CLK limit +"] + out_pos_idx
    out_clk_lim_minus = population.template.out_idx["CLK limit -"] + out_pos_idx
    out_rr_lim_plus = population.template.out_idx["RR limit +"] + out_pos_idx
    out_rr_lim_minus = population.template.out_idx["RR limit -"] + out_pos_idx
    out_rr_override = population.template.out_idx["RR OV"] + out_pos_idx
    out_thresh_plus = population.template.out_idx["THRESH +"] + out_pos_idx
    out_thresh_minus = population.template.out_idx["THRESH -"] + out_pos_idx

    if USE_GPU:
        if population.pop_count < 2000:
            threadsperblock = 8
        elif population.pop_count < 4000:
            threadsperblock = 16
        else:
            threadsperblock = 32
        blockspergrid = math.ceil(population.pop_count / threadsperblock)

        _decide_trader[blockspergrid, threadsperblock](population.d_O, population.d_pos, population.d_age,
                                                       population.d_alive,
                                                       population.d_clk_counter, population.d_clk_lim, population.d_clk_sig,
                                                       population.max_CLK_period,
                                                       population.d_rr_counter, population.d_rr_lim,
                                                       population.max_RR_period,
                                                       population.d_thresh,
                                                       out_pos_idx, out_clk_lim_plus, out_clk_lim_minus,
                                                       out_rr_lim_plus, out_rr_lim_minus, out_rr_override,
                                                       out_thresh_plus, out_thresh_minus
                                                       )
    else:
        decide_trader(pos=population.pos, alive=population.alive, clk_sig=population.clk_sig, age=population.age,
                      clk_counter=population.clk_counter, clk_lim=population.clk_lim, O=population.O,
                      dec_thresh=population.thresh, max_CLK_period=population.max_CLK_period,
                      rr_counter=population.rr_counter, rr_lim=population.rr_lim, max_rr_period=population.max_RR_period,
                      out_pos_idx=out_pos_idx, out_clk_lim_plus_idx=out_clk_lim_plus,
                      out_clk_lim_minus_idx=out_clk_lim_minus,
                      out_rr_lim_plus_idx=out_rr_lim_plus, out_rr_lim_minus_idx=out_rr_lim_minus, out_rr_override=out_rr_override,
                      out_thresh_plus_idx=out_thresh_plus, out_thresh_minus_idx=out_thresh_minus)


def calc_decide_gamer(population, USE_GPU=True):

    # Find output node indexes
    out_start_idx = population.out_offset
    out_pos_idx = out_start_idx

    if USE_GPU:
        if population.pop_count < 2000:
            threadsperblock = 8
        elif population.pop_count < 4000:
            threadsperblock = 16
        else:
            threadsperblock = 32
        blockspergrid = math.ceil(population.pop_count / threadsperblock)

        _decide_gamer[blockspergrid, threadsperblock](population.d_O, population.d_pos, population.d_alive,
                                                       population.decision_threshold, out_pos_idx)

def calc_decide_2d(population, sim_size, prey_population=None):
    if population.pop_count < 2000:
        threadsperblock = 8
    elif population.pop_count < 4000:
        threadsperblock = 16
    else:
        threadsperblock = 32
    blockspergrid = math.ceil(population.pop_count / threadsperblock)

    d_rand_xy = population.rand_movement_array(GPU=True)  # random array for random movement

    if population.template.name == "grazer":

        out_mov_up = population.template.decision_indexes['M-up'] + population.out_offset
        out_mov_dwn = population.template.decision_indexes['M-dwn'] + population.out_offset
        out_mov_rght = population.template.decision_indexes['M-rght'] + population.out_offset
        out_mov_lft = population.template.decision_indexes['M-lft'] + population.out_offset
        out_mov_rnd = population.template.decision_indexes['M-rnd'] + population.out_offset
        out_clk_lim_plus = population.template.decision_indexes['CLK-P+'] + population.out_offset
        out_clk_lim_minus = population.template.decision_indexes['CLK-P-'] + population.out_offset
        out_movspd_plus = population.template.decision_indexes['MVSPD+'] + population.out_offset
        out_movspd_minus = population.template.decision_indexes['MVSPD-'] + population.out_offset
        out_pher = population.template.decision_indexes['pher-rel'] + population.out_offset

        _decide_grazer[blockspergrid, threadsperblock](population.d_O, population.d_pos, population.d_age,
                                                       population.d_alive,
                                                       population.d_clk_counter, population.d_clk_lim,
                                                       population.d_clk_sig,
                                                       population.d_pher_rel, population.d_mv_speed,
                                                       sim_size[0], sim_size[1],
                                                       population.max_mov_speed,
                                                       population.max_CLK_period,
                                                       population.decision_threshold, d_rand_xy,
                                                       out_mov_up, out_mov_dwn, out_mov_rght, out_mov_lft, out_mov_rnd,
                                                       out_clk_lim_plus, out_clk_lim_minus,
                                                       out_pher, out_movspd_plus, out_movspd_minus
                                                       )

    elif population.template.name == "predator":

        out_mov_up = population.template.decision_indexes['M-up'] + population.out_offset
        out_mov_dwn = population.template.decision_indexes['M-dwn'] + population.out_offset
        out_mov_rght = population.template.decision_indexes['M-rght'] + population.out_offset
        out_mov_lft = population.template.decision_indexes['M-lft'] + population.out_offset
        out_mov_rnd = population.template.decision_indexes['M-rnd'] + population.out_offset
        out_clk_lim_plus = population.template.decision_indexes['CLK-P+'] + population.out_offset
        out_clk_lim_minus = population.template.decision_indexes['CLK-P-'] + population.out_offset
        out_movspd_plus = population.template.decision_indexes['MVSPD+'] + population.out_offset
        out_movspd_minus = population.template.decision_indexes['MVSPD-'] + population.out_offset
        out_pher = population.template.decision_indexes['pher-rel'] + population.out_offset

        _decide_predator[blockspergrid, threadsperblock](population.d_O, population.d_pos, population.d_age,
                                                         population.d_alive, population.d_health,
                                                         population.d_clk_counter, population.d_clk_lim,
                                                         population.d_clk_sig,
                                                         population.d_pher_rel, population.d_mv_speed,
                                                         population.d_prey_dir, population.d_prey_detected,
                                                         prey_population.d_alive, prey_population.d_health,
                                                         population.eat_radius,
                                                         sim_size[0], sim_size[1],
                                                         population.max_mov_speed,
                                                         population.max_CLK_period,
                                                         population.decision_threshold, d_rand_xy,
                                                         population.min_health, population.max_health,
                                                         out_mov_up, out_mov_dwn, out_mov_rght, out_mov_lft,
                                                         out_mov_rnd,
                                                         out_clk_lim_plus, out_clk_lim_minus,
                                                         out_pher, out_movspd_plus, out_movspd_minus
                                                         )


def calc_decide(population, sim_size, prey_population=None, sim_type="2D", USE_GPU=True):

    if sim_type == "2D":
        calc_decide_2d(population=population, sim_size=sim_size, prey_population=prey_population)
    elif sim_type == "numerical":
        calc_decide_numerical(population=population, USE_GPU=USE_GPU)
    elif sim_type == "game":
        calc_decide_gamer(population=population, USE_GPU=USE_GPU)

