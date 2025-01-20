import numpy as np
from numba import cuda, config
import math

config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


"""
Numba/CUDA kernels for Genetic Simulation Operations
"""


# **************  SENSE  **************
@cuda.jit
def _sense_grazer(O, pos, health, age, alive, clk_sig, nn_dist,
                  pred_detected, pred_dist,
                  field, field_dx, field_dy,
                  t, sim_width, sim_height, min_health, max_health,
                  pop_dens, pop_dens_dx, pop_dens_dy, pop_dens_ch,
                  pher, pher_dx, pher_dy, pher_ch,
                  in_pos_x, in_pos_y, in_dist_nr, in_clk, in_field, in_field_dx, in_field_dy,
                  in_age, in_health, in_pher, in_pher_dx, in_pher_dy,
                  in_pred_detected, in_pred_dir_x, in_pred_dir_y,
                  in_PD, in_PD_dx, in_PD_dy):
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    # Sense state inputs
    if i_org < pos.shape[0]:
        if alive[i_org] == 1:
            # x and y positions on 2D grid (reversed for 2D field grid axis)
            # NOTE: we have rotated the field 90 degrees from default to ensure that x, y notation is consistent
            # with array indexing
            x_pos_idx = int(pos[i_org, 0] + sim_width / 2)
            y_pos_idx = int(pos[i_org, 1] + sim_height / 2)
            field_val = field[x_pos_idx, y_pos_idx, t]

            #if field_val < -0.9:
            #    alive[i_org] = 0
            #if field_val <= 0:
            health[i_org] += field_val
            #else:
            #    health[i_org] += field_val / (1. + 1.*pop_dens[pop_dens_ch, x_pos_idx, y_pos_idx]*pop_dens[pop_dens_ch, x_pos_idx, y_pos_idx]*pop_dens[pop_dens_ch, x_pos_idx, y_pos_idx])

            if health[i_org] >= max_health:
                health[i_org] = max_health
            if health[i_org] < min_health:
                alive[i_org] = 0

            O[i_org, 0, in_pos_x] = pos[i_org, 0] / sim_width * 2
            O[i_org, 0, in_pos_y] = pos[i_org, 1] / sim_height * 2
            O[i_org, 0, in_dist_nr] = nn_dist[i_org]  # distance to nearest, leave empty for now
            O[i_org, 0, in_clk] = clk_sig[i_org]  # clk.sig
            O[i_org, 0, in_field] = field_val
            O[i_org, 0, in_field_dx] = field_dx[x_pos_idx, y_pos_idx, t]
            O[i_org, 0, in_field_dy] = field_dy[x_pos_idx, y_pos_idx, t]
            O[i_org, 0, in_health] = health[i_org]
            O[i_org, 0, in_age] = age[i_org]
            O[i_org, 0, in_pher] = pher[pher_ch, x_pos_idx, y_pos_idx]
            O[i_org, 0, in_pher_dx] = pher_dx[pher_ch, x_pos_idx, y_pos_idx]
            O[i_org, 0, in_pher_dy] = pher_dy[pher_ch, x_pos_idx, y_pos_idx]
            O[i_org, 0, in_pred_detected] = (pred_detected[i_org] >= 0)*1.
            O[i_org, 0, in_pred_dir_x] = -pred_dist[i_org, 0]
            O[i_org, 0, in_pred_dir_y] = -pred_dist[i_org, 1]
            #O[i_org, 0, in_PD] = pop_dens[pop_dens_ch, x_pos_idx, y_pos_idx]
            #O[i_org, 0, in_PD_dx] = pop_dens_dx[pop_dens_ch, x_pos_idx, y_pos_idx]
            #O[i_org, 0, in_PD_dy] = pop_dens_dy[pop_dens_ch, x_pos_idx, y_pos_idx]


@cuda.jit
def _sense_predator(O, pos, health, age, alive, clk_sig, nn_dist,
                    prey_detected, prey_dist,
                    field, field_dx, field_dy,
                    t, sim_width, sim_height,
                    pher, pher_dx, pher_dy, pher_ch,
                    in_pos_x, in_pos_y, in_dist_nr, in_clk, in_field, in_field_dx, in_field_dy,
                    in_age, in_health, in_pher, in_pher_dx, in_pher_dy,
                    in_prey_detected, in_prey_dir_x, in_prey_dir_y):
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    # Sense state inputs
    if i_org < pos.shape[0]:
        if alive[i_org] == 1:
            # x and y positions on 2D grid (reversed for 2D field grid axis)
            # NOTE: we have rotated the field 90 degrees from default to ensure that x, y notation is consistent
            # with array indexing
            x_pos_idx = int(pos[i_org, 0] + sim_width / 2)
            y_pos_idx = int(pos[i_org, 1] + sim_height / 2)
            field_val = field[x_pos_idx, y_pos_idx, t]

            O[i_org, 0, in_pos_x] = pos[i_org, 0] / sim_width * 2
            O[i_org, 0, in_pos_y] = pos[i_org, 1] / sim_height * 2
            O[i_org, 0, in_dist_nr] = nn_dist[i_org]  # distance to nearest neighbour
            O[i_org, 0, in_clk] = clk_sig[i_org]
            O[i_org, 0, in_field] = field_val
            O[i_org, 0, in_field_dx] = field_dx[x_pos_idx, y_pos_idx, t]
            O[i_org, 0, in_field_dy] = field_dy[x_pos_idx, y_pos_idx, t]
            O[i_org, 0, in_health] = health[i_org]
            O[i_org, 0, in_age] = age[i_org]
            O[i_org, 0, in_pher] = pher[pher_ch, x_pos_idx, y_pos_idx]
            O[i_org, 0, in_pher_dx] = pher_dx[pher_ch, x_pos_idx, y_pos_idx]
            O[i_org, 0, in_pher_dy] = pher_dy[pher_ch, x_pos_idx, y_pos_idx]
            O[i_org, 0, in_prey_detected] = (prey_detected[i_org] >= 0)*1.
            O[i_org, 0, in_prey_dir_x] = -prey_dist[i_org, 0]
            O[i_org, 0, in_prey_dir_y] = -prey_dist[i_org, 1]


@cuda.jit
def _sense_trader(O, pos, age, alive, clk_sig, data, pos_dens,
                  t, input_pos_idx, input_clk_idx, input_age_idx, input_data_idx, input_pos_density_idx):
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    # Sense state inputs
    if i_org < pos.shape[0]:
        if alive[i_org] == 1:
            '''            if health[i_org] >= max_health:
                            health[i_org] = max_health
                        if health[i_org] < min_health:
                            alive[i_org] = 0'''

            # Assign current position inputs
            for i_pos in range(pos.shape[1]):
                O[i_org, 0, input_pos_idx + i_pos] = pos[i_org, i_pos]

            O[i_org, 0, input_clk_idx] = clk_sig[i_org]  # CLK state input
            O[i_org, 0, input_age_idx] = age[i_org]/100  # Organism age input

            # assign data inputs
            for n in range(data.shape[0]):
                O[i_org, 0, input_data_idx+n] = data[n, t]

            # assign position density inputs
            # First assign all '1' position densities for all populations
            # 1-states
            no_pos_density_ch = pos_dens.shape[0]
            for ch in range(no_pos_density_ch):
                for n in range(pos_dens.shape[1]):
                    O[i_org, 0, input_pos_density_idx + ch*pos_dens.shape[1] + n] = pos_dens[ch, n]
            # 0-states
            for ch in range(no_pos_density_ch):
                for n in range(pos_dens.shape[1]):
                    O[i_org, 0, input_pos_density_idx + no_pos_density_ch*pos_dens.shape[1] + ch*pos_dens.shape[1] + n] = 1. - pos_dens[ch, n]



@cuda.jit
def _sense_gamer(O, pos, age, alive, clk_sig, actb, data, pos_dens,
                  input_pos_idx, input_acnt_idx, input_data_idx, input_pos_density_idx):
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    # Sense state inputs
    if i_org < pos.shape[0]:
        if alive[i_org] == 1:
            '''            if health[i_org] >= max_health:
                            health[i_org] = max_health
                        if health[i_org] < min_health:
                            alive[i_org] = 0'''

            # Assign current position inputs
            for i_pos in range(pos.shape[1]):
                O[i_org, 0, input_pos_idx + i_pos] = pos[i_org, i_pos]

            O[i_org, 0, input_acnt_idx] = actb[i_org]  # Organism accountability score

            # assign data inputs
            for n in range(data.shape[0]):
                O[i_org, 0, input_data_idx+n] = data[n]

            # assign position density inputs
            no_pos_density_ch = pos_dens.shape[0]
            for ch in range(no_pos_density_ch):
                for n in range(pos_dens.shape[1]):
                    O[i_org, 0, input_pos_density_idx + ch*pos_dens.shape[1] + n] = pos_dens[ch, n]
            # 0-states
            for ch in range(no_pos_density_ch):
                for n in range(pos_dens.shape[1]):
                    O[i_org, 0, input_pos_density_idx + no_pos_density_ch*pos_dens.shape[1] + ch*pos_dens.shape[1] + n] = 1. - pos_dens[ch, n]


#  **************  DECIDE  **************
@cuda.jit
def _decide_grazer(O, pos, age, alive, clk_counter, clk_lim, clk_sig, pher_rel, mv_speed,
                   sim_width, sim_height,
                   max_mov_speed, max_CLK_period,
                   dec_thresh,
                   pos_rand,
                   out_mov_up, out_mov_dwn, out_mov_rght, out_mov_lft, out_mov_rnd, out_clk_lim_plus, out_clk_lim_minus,
                   out_pher, out_movspd_plus, out_movspd_minus):
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    if i_org < pos.shape[0]:
        if alive[i_org] == 1:
            age[i_org] += 1
            clk_counter[i_org] += 1
            if clk_counter[i_org] > clk_lim[i_org]:
                if clk_sig[i_org] == 1:
                    clk_sig[i_org] = 0
                else:
                    clk_sig[i_org] = 1
                clk_counter[i_org] = 0

            if O[i_org, 0, out_mov_rnd] > dec_thresh:  # move random
                pos[i_org, 0] += int(pos_rand[i_org, 0] * 2 - 1)
                pos[i_org, 1] += int(pos_rand[i_org, 1] * 2 - 1)
            else:
                if O[i_org, 0, out_mov_up] > dec_thresh:
                    pos[i_org, 1] += mv_speed[i_org]  # move up
                elif O[i_org, 0, out_mov_dwn] > dec_thresh:
                    pos[i_org, 1] -= mv_speed[i_org]  # move down
                if O[i_org, 0, out_mov_rght] > dec_thresh:
                    pos[i_org, 0] += mv_speed[i_org]  # move right
                elif O[i_org, 0, out_mov_lft] > dec_thresh:
                    pos[i_org, 0] -= mv_speed[i_org]  # move left
            if pos[i_org, 0] >= sim_width / 2:
                pos[i_org, 0] = sim_width / 2 - 1
            elif pos[i_org, 0] < -sim_width / 2:
                pos[i_org, 0] = -sim_width / 2
            if pos[i_org, 1] >= sim_height / 2:
                pos[i_org, 1] = sim_height / 2 - 1
            elif pos[i_org, 1] < -sim_height / 2:
                pos[i_org, 1] = -sim_height / 2

            if O[i_org, 0, out_clk_lim_plus] > dec_thresh:  # increase clock period
                clk_lim[i_org] += 1
                if clk_lim[i_org] > max_CLK_period:
                    clk_lim[i_org] = max_CLK_period
            if O[i_org, 0, out_clk_lim_minus] > dec_thresh:  # decrease clock period
                clk_lim[i_org] -= 1
                if clk_lim[i_org] < 1:
                    clk_lim[i_org] = 1

            # pheromone control
            if O[i_org, 0, out_pher] > dec_thresh:
                pher_rel[i_org] = 0.2
            else:
                pher_rel[i_org] = 0.

            # speed adjustment
            if O[i_org, 0, out_movspd_plus] > dec_thresh:
                mv_speed[i_org] += 1
            elif O[i_org, 0, out_movspd_minus] > dec_thresh:
                mv_speed[i_org] -= 1
            if mv_speed[i_org] > max_mov_speed:
                mv_speed[i_org] = max_mov_speed
            elif mv_speed[i_org] < 1:
                mv_speed[i_org] = 1


@cuda.jit
def _decide_predator(O, pos, age, alive, health, clk_counter, clk_lim, clk_sig, pher_rel, mv_speed,
                     prey_dir, prey_detected, prey_alive, prey_health, eat_radius,
                     max_mov_speed, max_CLK_period,
                     sim_width, sim_height, dec_thresh,
                     pos_rand,
                     min_health, max_health,
                     out_mov_up, out_mov_dwn, out_mov_rght, out_mov_lft, out_mov_rnd, out_clk_lim_plus,
                     out_clk_lim_minus,
                     out_pher, out_movspd_plus, out_movspd_minus):

    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    if i_org < pos.shape[0]:
        if alive[i_org] == 1:
            age[i_org] += 1
            clk_counter[i_org] += 1
            if clk_counter[i_org] > clk_lim[i_org]:
                if clk_sig[i_org] == 1:
                    clk_sig[i_org] = 0
                else:
                    clk_sig[i_org] = 1
                clk_counter[i_org] = 0

            if O[i_org, 0, out_mov_rnd] > dec_thresh:  # move random
                pos[i_org, 0] += int(pos_rand[i_org, 0] * 2 - 1)
                pos[i_org, 1] += int(pos_rand[i_org, 1] * 2 - 1)
            else:
                if O[i_org, 0, out_mov_up] > dec_thresh:
                    pos[i_org, 1] += mv_speed[i_org]  # move up
                elif O[i_org, 0, out_mov_dwn] > dec_thresh:
                    pos[i_org, 1] -= mv_speed[i_org]  # move down
                if O[i_org, 0, out_mov_rght] > dec_thresh:
                    pos[i_org, 0] += mv_speed[i_org]  # move right
                elif O[i_org, 0, out_mov_lft] > dec_thresh:
                    pos[i_org, 0] -= mv_speed[i_org]  # move left
            if pos[i_org, 0] >= sim_width / 2:
                pos[i_org, 0] = sim_width / 2 - 1
            elif pos[i_org, 0] < -sim_width / 2:
                pos[i_org, 0] = -sim_width / 2
            if pos[i_org, 1] >= sim_height / 2:
                pos[i_org, 1] = sim_height / 2 - 1
            elif pos[i_org, 1] < -sim_height / 2:
                pos[i_org, 1] = -sim_height / 2

            if O[i_org, 0, out_clk_lim_plus] > dec_thresh:  # increase clock period
                clk_lim[i_org] += 1
                if clk_lim[i_org] > max_CLK_period:
                    clk_lim[i_org] = max_CLK_period
            if O[i_org, 0, out_clk_lim_minus] > dec_thresh:  # decrease clock period
                clk_lim[i_org] -= 1
                if clk_lim[i_org] < 1:
                    clk_lim[i_org] = 1

            # check consume prey
            if prey_detected[i_org] >= 0:  # prey within vision radius
                if prey_dir[i_org, 2] <= eat_radius:  # prey within eat radius
                    if prey_alive[prey_detected[i_org]] == 1:  # target prey still alive
                        health[i_org] += 1
                        prey_alive[prey_detected[i_org]] = 0
                        prey_health[prey_detected[i_org]] = prey_health[prey_detected[i_org]] - 100

            # pheromone control
            if O[i_org, 0, out_pher] > dec_thresh:
                pher_rel[i_org] = 0.2
            else:
                pher_rel[i_org] = 0.

            # speed adjustment
            if O[i_org, 0, out_movspd_plus] > dec_thresh:
                mv_speed[i_org] += 1
            elif O[i_org, 0, out_movspd_minus] > dec_thresh:
                mv_speed[i_org] -= 1
            if mv_speed[i_org] > max_mov_speed:
                mv_speed[i_org] = max_mov_speed
            elif mv_speed[i_org] < 1:
                mv_speed[i_org] = 1

            if health[i_org] >= max_health:
                health[i_org] = max_health
            if health[i_org] < min_health:
                alive[i_org] = 0


@cuda.jit
def _decide_trader(O, pos, age, alive,
                   clk_counter, clk_lim, clk_sig, max_CLK_period,
                   rr_counter, rr_lim, max_rr_period,
                   dec_thresh,
                   out_pos_idx, out_clk_lim_plus_idx, out_clk_lim_minus_idx,
                   out_rr_lim_plus_idx, out_rr_lim_minus_idx, out_rr_override,
                   out_thresh_plus_idx, out_thresh_minus_idx):

    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    if i_org < pos.shape[0]:
        if alive[i_org] == 1:
            current_thresh = dec_thresh[i_org]

            age[i_org] += 1
            clk_counter[i_org] += 1
            if clk_counter[i_org] > clk_lim[i_org]:
                clk_sig[i_org] = 1
                clk_counter[i_org] = 0
            else:
                clk_sig[i_org] = 0

            reaction = 0
            reaction_override = 0
            # Check reaction override
            if O[i_org, 0, out_rr_override] > current_thresh:
                reaction_override = 1

            rr_counter[i_org] += 1
            if rr_counter[i_org] >= rr_lim[i_org] or reaction_override == 1:
                rr_counter[i_org] = 0
                reaction = 1

            sum_zero = True
            for i_pos in range(pos.shape[1]):
                if O[i_org, 0, out_pos_idx + i_pos] > current_thresh:
                    sum_zero = False
            if not sum_zero and reaction:
                for i_pos in range(pos.shape[1]):
                    pos[i_org, i_pos] = 0.
                    if O[i_org, 0, out_pos_idx + i_pos] > current_thresh:
                        pos[i_org, i_pos] = 1.

            '''
                        ### ******* When sum==0 >> only USDC = 1
                        # Decide buy/sell position
                        for i_pos in range(pos.shape[1]):
                            pos[i_org, i_pos] = 0.
                            if O[i_org, 0, out_pos_idx + i_pos] > dec_thresh:
                                pos[i_org, i_pos] = 1.
                                sum_zero = False
                        # if all positions zero, set full position to USD (last position index)
                        if sum_zero:
                            pos[i_org, pos.shape[1]-1] = 1.
            '''


            if O[i_org, 0, out_clk_lim_plus_idx] > current_thresh:  # increase clock period
                clk_lim[i_org] += 1
                if clk_lim[i_org] > max_CLK_period:
                    clk_lim[i_org] = max_CLK_period
            if O[i_org, 0, out_clk_lim_minus_idx] > current_thresh:  # decrease clock period
                clk_lim[i_org] -= 1
                if clk_lim[i_org] < 1:
                    clk_lim[i_org] = 1
            if O[i_org, 0, out_rr_lim_plus_idx] > current_thresh:  # increase reaction rate period
                rr_lim[i_org] += 1
                if rr_lim[i_org] > max_rr_period:
                    rr_lim[i_org] = max_rr_period
            if O[i_org, 0, out_rr_lim_minus_idx] > current_thresh:  # decrease reaction rate period
                rr_lim[i_org] -= 1
                if rr_lim[i_org] < 1:
                    rr_lim[i_org] = 1
            if O[i_org, 0, out_thresh_plus_idx] > current_thresh:  # increase decision_threshold
                dec_thresh[i_org] += 0.02
                if dec_thresh[i_org] > 0.98:
                    dec_thresh[i_org] = 0.98
            if O[i_org, 0, out_thresh_minus_idx] > current_thresh:  # decrease decision_threshold
                dec_thresh[i_org] -= 0.02
                if dec_thresh[i_org] < 0.1:
                    dec_thresh[i_org] = 0.1


@cuda.jit
def _decide_gamer(O, pos, alive, dec_thresh, out_pos_idx):

    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    if i_org < pos.shape[0]:
        if alive[i_org] == 1:
            sum_zero = True
            for i_pos in range(pos.shape[1]):
                if O[i_org, 0, out_pos_idx + i_pos] > dec_thresh:
                    sum_zero = False
            if not sum_zero:
                for i_pos in range(pos.shape[1]):
                    pos[i_org, i_pos] = 0.
                    if O[i_org, 0, out_pos_idx + i_pos] > dec_thresh:
                        pos[i_org, i_pos] = 1.


#  **************  Get Position Change Count  **************
@cuda.jit
def _position_change(pos, pos_last, pos_change):
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    if i_org < pos.shape[0]:
        changed = 0.
        total_pos = 0.
        for i_pos in range(pos.shape[1]):
            total_pos += pos[i_org, i_pos] + pos_last[i_org, i_pos]
            if not pos[i_org, i_pos] == pos_last[i_org, i_pos]:
                changed += 1.
                #changed = 1
                #break
        pos_change[i_org] += changed/total_pos


#  **************  Position Mask  **************
@cuda.jit
def _apply_position_mask(pos, pos_mask):
    """
    Given a position mask consisting of "1" (active) and "0" inactive, apply this mask to each organism's positions
    to deactivate positions which are no longer available.
    :param pos:
    :param pos_mask:
    :return:
    """
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    if i_org < pos.shape[0]:
        pos_sum = 0
        for i_pos in range(pos.shape[1]):
            if pos_mask[i_pos] == 0:
                pos[i_org, i_pos] = 0.
            else:
                pos_sum += pos[i_org, i_pos]
        # Ensure new positions are not all 0
        if pos_sum == 0:
            pos[i_org, pos.shape[1]-1] = 1.

# **************  FITNESS  **************
@cuda.jit
def _vector_fitness(fitness, vec):
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    # Sense state inputs
    if i_org < fitness.shape[0]:
        fitness[i_org] = vec[i_org]


@cuda.jit
def _field_fitness(fitness, pos, field, t, sim_width, sim_height):
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    # Sense state inputs
    if i_org < fitness.shape[0]:
        x_pos_idx = int(pos[i_org, 0] + sim_width / 2)
        y_pos_idx = int(pos[i_org, 1] + sim_height / 2)
        field_val = field[x_pos_idx, y_pos_idx, t]
        fitness[i_org] += field_val


@cuda.jit
def _roi_fitness2(fitness, pos, P, t, sim_width, sim_height):
    """
    ROI fitness function for two currencies including the baseline (c_0=1)
    p_0 = pos[0]/sim_width
    y-direction is unused
    Fitness = current value = F_t = F_(t-1) * sum_i(c_i * p_i)
    :param fitness:
    :param pos:
    :param C: array of currency prices [N-1, T] (in this case N=2)
    :param t: current time index
    :param sim_width:
    :param sim_height:
    :return:
    """
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    # Sense state inputs
    if i_org < fitness.shape[0]:
        g_0 = (pos[i_org, 0] + sim_width / 2) / sim_width  # usd
        g_1 = 1. - g_0  # btc

        fitness[i_org] = fitness[i_org] * (g_0 + P[0, t + 1] / P[0, t] * g_1)


@cuda.jit
def _roi_fitness4(fitness, pos, P, pos_map, t, sim_width, sim_height):
    """
    ROI fitness function for four currencies including the baseline (c_0=1)
    p_0 = [pos[0]/sim_width, pos[1]/sim_height]
    Fitness = current value = F_t = F_(t-1) * sum_i(c_i * p_i)
    :param fitness:
    :param pos:
    :param C: array of currency prices [N-1, T] (in this case N=4)
    :param t: current time index
    :param sim_width:
    :param sim_height:
    :return:
    """
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    # Sense state inputs
    if i_org < fitness.shape[0]:
        x_pos_idx = int(pos[i_org, 0] + sim_width / 2)
        y_pos_idx = int(pos[i_org, 1] + sim_height / 2)

        g_0 = pos_map[x_pos_idx, y_pos_idx, 0]
        g_1 = pos_map[x_pos_idx, y_pos_idx, 1]
        g_2 = pos_map[x_pos_idx, y_pos_idx, 2]
        g_3 = pos_map[x_pos_idx, y_pos_idx, 3]

        # OPTIMIZATION: We can precalculate P[t+1]/P[t]
        fitness[i_org] = fitness[i_org] * (g_3 + P[0, t + 1] / P[0, t] * g_1 + P[1, t + 1] / P[1, t] * g_2 +
                                           P[2, t + 1] / P[2, t] * g_0)


@cuda.jit
def _roi_fitness(fitness, pos, pos_last, data, spread_ratio, t, fee_pct, use_spread, use_fee, loser_fit):
    """
    ROI fitness function for four currencies including the baseline (c_0=1). Include fees associated with the ask/bid
    spread rate.
    Fitness = current value = F_t = F_(t-1) * sum_i(c_i * p_i)
    :param fitness:
    :param pos:
    :param data: array of currency prices [N-1, T]
    :param spread ratio: the spread ratio (spread_percent/100). Array with length [N-1]
    :param t: current time index
    :param fee_pct: array of trading fees, one for each currency (in percent)
    :return:
    """
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    # Sense state inputs
    if i_org < fitness.shape[0]:
        pos_sum = 0.
        for i_pos in range(pos.shape[1]):
            pos_sum += pos[i_org, i_pos]

        sum_last = 0.
        for i_pos in range(pos.shape[1]):
            sum_last += pos_last[i_org, i_pos]

        if sum_last == 0:
            sum_last = 1

        # Calculate ROI
        if pos_sum == 0:  # all positions 0. Set to USD
            roi = 1.
        else:
            #price_avg_current = 0.
            #price_avg_next = 0.
            roi = 0.
            roi_fee = 0.
            for i_pos in range(pos.shape[1]):
                roi = roi + pos[i_org, i_pos] / pos_sum * data[i_pos, t + 1] / data[i_pos, t]
                #price_avg_current += data[i_pos, t] * pos[i_org, i_pos] / pos_sum
                #price_avg_next += data[i_pos, t + 1] * pos[i_org, i_pos] / pos_sum
                #price_avg_current += data[i_pos, t] * pos[i_org, i_pos]
                #price_avg_next += data[i_pos, t + 1] * pos[i_org, i_pos]

                if use_spread == 2:  # Calculate spread based on average spread float
                    if not i_pos == (pos.shape[1]-1):  # ignore spread calculation on base currency
                        roi_fee = roi_fee
                        '''if pos[i_org, i_pos] > pos_last[i_org, i_pos]:  # buy order spread fee
                            roi_fee = roi_fee - spread_ratio[i_pos]/(1+spread_ratio[i_pos]) *\
                                  math.fabs(pos[i_org, i_pos]/pos_sum - pos_last[i_org, i_pos]/sum_last)
                        else:  # sell order spread fee
                            roi_fee = roi_fee - spread_ratio[i_pos] * \
                                  math.fabs(pos_last[i_org, i_pos] / sum_last - pos[i_org, i_pos] / pos_sum)'''
                elif use_spread == 1:  # Calculate spread based on detailed spread array
                    if not i_pos == (pos.shape[1] - 1):  # ignore spread calculation on base currency
                        if pos[i_org, i_pos] > pos_last[i_org, i_pos]:  # buy order spread fee
                            roi_fee = roi_fee - spread_ratio[i_pos, t] / (1 + spread_ratio[i_pos, t]) * \
                                  math.fabs(pos[i_org, i_pos] / pos_sum - pos_last[i_org, i_pos] / sum_last)
                        else:  # sell order spread fee
                            roi_fee = roi_fee - spread_ratio[i_pos, t] * \
                                  math.fabs(pos_last[i_org, i_pos] / sum_last - pos[i_org, i_pos] / sum_last    )
                else:
                    roi_fee = roi_fee
                if use_fee:  # Calculate fee
                    roi_fee = roi_fee - 0.5 * fee_pct[i_pos] / 100.0 * math.fabs(pos[i_org, i_pos] / pos_sum - pos_last[i_org, i_pos] / sum_last)
            if loser_fit == 1:
                roi = 1./roi + roi_fee
                #roi = price_avg_current / price_avg_next + roi_fee
            else:
                roi = roi + roi_fee
                #roi = price_avg_next / price_avg_current + roi_fee

        fitness[i_org] = fitness[i_org] * roi


@cuda.jit
def _roi_fitness_loser_normalized(fitness, pos, data, t):
    """
    ROI fitness function for four currencies including the baseline (c_0=1)
    p_0 = [pos[0]/sim_width, pos[1]/sim_height]
    Fitness = current value = F_t = F_(t-1) * sum_i(c_i * p_i)
    :param fitness:
    :param pos:
    :param data: array of currency prices [N-1, T]
    :param t: current time index
    :return:
    """
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    # Sense state inputs
    if i_org < fitness.shape[0]:

        sum = 0.
        for i_pos in range(pos.shape[1]):
            sum += pos[i_org, i_pos]

        # normalize positions
        #for i_pos in range(pos.shape[1]):
        #    pos[i_org, i_pos] = math.floor(pos[i_org, i_pos]/sum*100)/100

        # ensure sum is =1, if not, add extra to USD
        '''        sum = 0.
                for i_pos in range(pos.shape[1]):
                    sum += pos[i_org, i_pos]
                pos[i_org, -1] += 1. - sum'''

        roi = 0.
        if sum == 0:  # all positions 0. Set to USD
            roi = 1.
        else:
            for i_pos in range(pos.shape[1]):
                roi = roi + pos[i_org, i_pos] / sum * data[i_pos, t] / data[i_pos, t+1]

        fitness[i_org] = fitness[i_org] * roi


@cuda.jit
def _roi_fitness_stabilizer_normalized(fitness, pos, data, t, sensitivity):
    """
    ROI fitness function for four currencies including the baseline (c_0=1)
    p_0 = [pos[0]/sim_width, pos[1]/sim_height]
    Fitness = current value = F_t = F_(t-1) * sum_i(c_i * p_i)
    :param fitness:
    :param pos:
    :param data: array of currency prices [N-1, T]
    :param t: current time index
    :return:
    """
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    # Sense state inputs
    if i_org < fitness.shape[0]:

        sum = 0.
        for i_pos in range(pos.shape[1]):
            sum += pos[i_org, i_pos]

        # normalize positions
        for i_pos in range(pos.shape[1]):
            pos[i_org, i_pos] = math.floor(pos[i_org, i_pos]/sum*100)/100

        # ensure sum is =1, if not, add extra to USD
        sum = 0.
        for i_pos in range(pos.shape[1]):
            sum += pos[i_org, i_pos]

        pos[i_org, -1] += 1. - sum

        roi = 0.
        if sum == 0:  # all positions 0. Set to USD
            roi = 1.
        else:
            for i_pos in range(pos.shape[1]):
                roi = roi + pos[i_org, i_pos] * data[i_pos, t] / data[i_pos, t+1]

        fitness[i_org] = fitness[i_org] * 1 / ((roi-1)**2 * sensitivity + 1)

@cuda.jit
def _roi_fitness_destabilizer_normalized(fitness, pos, data, t, sensitivity, offset):
    """
    ROI fitness function for four currencies including the baseline (c_0=1)
    p_0 = [pos[0]/sim_width, pos[1]/sim_height]
    Fitness = current value = F_t = F_(t-1) * sum_i(c_i * p_i)
    :param fitness:
    :param pos:
    :param data: array of currency prices [N-1, T]
    :param t: current time index
    :return:
    """
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    # Sense state inputs
    if i_org < fitness.shape[0]:

        sum = 0.
        for i_pos in range(pos.shape[1]):
            sum += pos[i_org, i_pos]

        # normalize positions
        for i_pos in range(pos.shape[1]):
            pos[i_org, i_pos] = math.floor(pos[i_org, i_pos]/sum*100)/100

        # ensure sum is =1, if not, add extra to USD
        sum = 0.
        for i_pos in range(pos.shape[1]):
            sum += pos[i_org, i_pos]

        pos[i_org, -1] += 1. - sum

        roi = 0.
        if sum == 0:  # all positions 0. Set to USD
            roi = 1.
        else:
            for i_pos in range(pos.shape[1]):
                roi = roi + pos[i_org, i_pos] * data[i_pos, t] / data[i_pos, t+1]

        fitness[i_org] = fitness[i_org] * sensitivity * ((roi-1)**2 + offset)


# ************** Reaction-Position Update ************
@cuda.jit
def _update_position(O, position_count, dec_thresh, input_pos_idx, out_pos_idx):
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    # Sense state inputs
    if i_org < O.shape[0]:
        sum_zero = True
        for i_pos in range(position_count):
            if O[i_org, 0, out_pos_idx + i_pos] > dec_thresh:
                sum_zero = False  # If sum=0, keep the current position inputs
        if not sum_zero:
            for i_pos in range(position_count):
                O[i_org, 0, input_pos_idx + i_pos] = 0.
                if O[i_org, 0, out_pos_idx + i_pos] > dec_thresh:
                    O[i_org, 0, input_pos_idx + i_pos] = 1.  # assign position input based on current output


# **************  Pheromone  **************
@cuda.jit
def _calc_pheromone_2d(pher, pos, pher_amt, sim_width, sim_height, max_radius, min_radius, saturation_level, ch):
    x, y = cuda.grid(2)
    if x < pher.shape[1] and y < pher.shape[2]:
        for i_org in range(pos.shape[0]):
            grid_pos_x = x - sim_width / 2
            grid_pos_y = y - sim_height / 2
            dx = math.fabs(pos[i_org, 0] - grid_pos_x)
            if dx < max_radius:
                dy = math.fabs(pos[i_org, 1] - grid_pos_y)
                if dy < max_radius:
                    rad_dist = math.sqrt(dx * dx + dy * dy)
                    if rad_dist < max_radius:
                        if rad_dist > min_radius:
                            pher[ch, x, y] += pher_amt[i_org] / (rad_dist + 0.2)
        if pher[ch, x, y] > saturation_level:
            pher[ch, x, y] = saturation_level


# **************  Environment  **************
@cuda.jit
def _find_closest_2d(coord, closest_idx, closest_dist):
    """
    For each point, find the closest other point
    :param coord: list of [X,Y] coordinates
    :param vert_count: (int) number of points (length of coord)
    :param closest_idx: output array of size [vert_count] where the ith entry represent the index for the closest point
        coordinate to point i
    :return:
    """
    # Each thread finds the closest match for a single vertex 'pos' with all other vertices
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    pos = tx + bx * bw  # Compute flattened index inside the array

    closest_dist_temp = 100000
    closest_vert = -1

    if pos < closest_idx.shape[0]:  # Check array boundaries
        for i in range(closest_idx.shape[0]):
            if not i == pos:
                diff_x = coord[pos, 0] - coord[i, 0]
                diff_y = coord[pos, 1] - coord[i, 1]
                dist = math.sqrt(diff_x * diff_x + diff_y * diff_y)
                if dist < closest_dist_temp:
                    closest_vert = i
                    closest_dist_temp = dist
        closest_idx[pos] = closest_vert
        closest_dist[pos] = closest_dist_temp


@cuda.jit
def _find_closest_2d_species(coord_a, coord_b, closest_idx, closest_dist):
    """
    For each coordinate in coord_a, find the closest coordinate in coord_b
    :param coord: list of [X,Y] coordinates
    :param vert_count: (int) number of points (length of coord)
    :param closest_idx: output array of size [vert_count] where the ith entry represent the index for the closest point
        coordinate to point i
    :return:
    """
    # Each thread finds the closest match for a single coordinate in coord_a
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    pos = tx + bx * bw  # Compute flattened index inside the array

    closest_dist_temp = 100000
    closest_vert = -1

    if pos < closest_idx.shape[0]:  # Check array boundaries
        for i in range(coord_b.shape[0]):
            diff_x = coord_a[pos, 0] - coord_b[i, 0]
            diff_y = coord_a[pos, 1] - coord_b[i, 1]
            dist = math.sqrt(diff_x * diff_x + diff_y * diff_y)
            if dist < closest_dist_temp:
                closest_vert = i
                closest_dist_temp = dist
        closest_idx[pos] = closest_vert
        closest_dist[pos] = closest_dist_temp


@cuda.jit
def _find_closest_2d_prey_directional(coord_a, coord_b, alive_mask, closest_within_rad, dist, sensory_rad):
    """
    For each coordinate in coord_a, find the closest coordinate in coord_b. If it's within the sensory radius,
    calculate a normalizes (-1,1) [x,y] displacement
    :param coord_a: list of [X,Y] predator coordinates
    :param coord_b: list of [X,Y] prey coordinates
    :param closest_within_rad: array of 0 and 1 specifying if prey is within sensory radius
    :param dist: array of [N, 2] with values ranging from (-1,1) showing the direction of nearest prey
    :param sensory_rad: float specifying the maximum sensing distance of the predator
    :return:
    """
    # Each thread finds the closest match for a single coordinate in coord_a
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    pos = tx + bx * bw  # Compute flattened index inside the array

    closest_dist_temp = 100000
    closest_dist_x = 10000
    closest_dist_y = 10000
    closest_vert = -1

    if pos < dist.shape[0]:  # Check array boundaries
        for i in range(coord_b.shape[0]):
            if alive_mask[i] == 1:
                diff_x = coord_a[pos, 0] - coord_b[i, 0]
                diff_y = coord_a[pos, 1] - coord_b[i, 1]
                dist_temp = math.sqrt(diff_x * diff_x + diff_y * diff_y)
                if dist_temp < closest_dist_temp:
                    closest_vert = i
                    closest_dist_temp = dist_temp
                    closest_dist_x = diff_x
                    closest_dist_y = diff_y
        if closest_dist_temp < sensory_rad:
            # dist[pos, 0] = closest_dist_x/sensory_rad
            # dist[pos, 1] = closest_dist_y/sensory_rad
            dist[pos, 0] = closest_dist_x / 4.
            dist[pos, 1] = closest_dist_y / 4.
            dist[pos, 2] = closest_dist_temp
            closest_within_rad[pos] = closest_vert
        else:
            dist[pos, 0] = 0.
            dist[pos, 1] = 0.
            dist[pos, 2] = 0.
            closest_within_rad[pos] = -1


@cuda.jit
def _calc_population_density(PD, pos, sim_width, sim_height, saturation_level, density_multiplier, ch):
    x, y = cuda.grid(2)
    if x < PD.shape[1] and y < PD.shape[2]:
        for i_org in range(pos.shape[0]):
            grid_pos_x = x - sim_width / 2
            grid_pos_y = y - sim_height / 2
            dx = math.fabs(pos[i_org, 0] - grid_pos_x)
            dy = math.fabs(pos[i_org, 1] - grid_pos_y)
            rad_dist = math.sqrt(dx * dx + dy * dy)
            PD[ch, x, y] += 1. / (rad_dist + 0.1) * density_multiplier
        if PD[ch, x, y] > saturation_level:
            PD[ch, x, y] = saturation_level


@cuda.jit
def _calc_grid_field_ndim(grid, pos, signal, field, channel, normalize_factor):
    """
    Each thread calculated the distance-weighted field at a particular grid coordinate based on nearby points in the
    n-D space.
    :param grid:
    :param pos:
    :param sim_width:
    :param sim_height:
    :param saturation_level:
    :param density_multiplier:
    :param ch:
    :return:
    """

    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    ig = tx + bx * bw  # Compute flattened index inside the array

    no_grid_points, no_dims = grid.shape
    no_org = pos.shape[0]

    # For each grid coordinate:
    if ig < no_grid_points:  # Check grid coordinate index within bounds.
        field[channel, ig] = 0
        for i_org in range(no_org):  # For each organism
            dist_temp = 0
            for i_dim in range(no_dims):  # Calculate distance along each grid coordinate
                dist_temp += (grid[ig, i_dim] - pos[i_org, i_dim])**2
                if dist_temp > 0.3:
                    break
            if dist_temp < 0.3:
                dist_temp = math.sqrt(dist_temp)  # get the Euclidean distance
                field[channel, ig] += signal[i_org]/(dist_temp+1)  # Calculate the signal-distance weighted field at the grid coordinate
        field[channel, ig] = field[channel, ig]/normalize_factor


@cuda.jit
def _calc_apply_field_ndim(grid, pos, field, values):
    """
    For each point, find the closest other point in n-Dimensions and assign the associated field value for that other
    point.
    :return:
    """
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    no_grid_points, no_dims = grid.shape
    no_org = pos.shape[0]

    # For each organism:
    if i_org < no_org:  # Check array boundaries
        values[i_org] = -1
        dist_nearest = 999
        for ig in range(no_grid_points):  # For each grid location
            dist_temp = 0
            for i_dim in range(no_dims):  # Calculate distance along each grid coordinate
                dist_temp += (grid[ig, i_dim] - pos[i_org, i_dim])**2
                if dist_temp > 0.15:
                    break
            #dist_temp = math.sqrt(dist_temp)  # get the Euclidean distance
            if dist_temp < dist_nearest:
                dist_nearest = dist_temp
                values[i_org] = field[ig]


@cuda.jit
def _calc_grid_find_closest_ndim(grid, pos, closest_idx):
    """
    For each point, find the closest other point in n-Dimensions and assign the associated field value for that other
    point.
    :return:
    """
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    no_grid_points, no_dims = grid.shape
    no_org = pos.shape[0]

    # For each organism:
    if i_org < no_org:  # Check array boundaries
        closest_idx[i_org] = -1
        dist_nearest = 999
        for ig in range(no_grid_points):  # For each grid location
            dist_temp = 0
            for i_dim in range(no_dims):  # Calculate distance along each grid coordinate
                dist_temp += (grid[ig, i_dim] - pos[i_org, i_dim])**2
                if dist_temp > 0.15:
                    break
            #dist_temp = math.sqrt(dist_temp)  # get the Euclidean distance
            if dist_temp < dist_nearest:
                dist_nearest = dist_temp
                closest_idx[i_org] = ig


@cuda.jit
def _calc_apply_field_with_index_ndim(pos, field, closest_idx, field_values, channel):
    """
    For each point, find the closest other point in n-Dimensions and assign the associated field value for that other
    point.
    :return:
    """
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    bx = cuda.blockIdx.x  # Block id in a 1D grid
    bw = cuda.blockDim.x  # Block width, i.e. number of threads per block
    i_org = tx + bx * bw  # Compute flattened index inside the array

    #no_grid_points, no_dims = grid.shape
    no_org = pos.shape[0]

    # For each organism:
    if i_org < no_org:  # Check array boundaries
        field_values[i_org] = field[channel, closest_idx[i_org]]


