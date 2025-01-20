import numpy as np
from numba import jit
import math

"""
Numba/CUDA kernels for Genetic Simulation Operations
"""


# **************  SENSE  **************
def sense_trader(pos, alive, clk_sig, age, data, pos_dens, pos_dens_ch, O, t,
                 input_pos_idx, input_clk_idx, input_age_idx, input_data_idx, input_pos_density_idx):

    no_org = O.shape[0]

    for i_org in range(no_org):
        if alive[i_org] == 1:

            # Assign current position inputs
            for i_pos in range(pos.shape[1]):
                O[i_org, 0, input_pos_idx + i_pos] = pos[i_org, i_pos]
            O[i_org, 0, input_clk_idx] = clk_sig[i_org]  # CLK state input
            O[i_org, 0, input_age_idx] = age[i_org]/100  # Organism age input

            # assign data inputs
            for n in range(data.shape[0]):
                O[i_org, 0, input_data_idx+n] = data[n, t]

            # assign position density inputs
            no_pos_density_ch = pos_dens.shape[0]
            for ch in range(no_pos_density_ch):
                for n in range(pos_dens.shape[1]):
                    O[i_org, 0, input_pos_density_idx + ch*pos_dens.shape[1] + n] = pos_dens[ch, n]
                    #O[i_org, 0, no_static_inputs + no_data_inputs + n] = 0.
            # 0-states
            for ch in range(no_pos_density_ch):
                for n in range(pos_dens.shape[1]):
                    O[i_org, 0, input_pos_density_idx + no_pos_density_ch*pos_dens.shape[1] + ch*pos_dens.shape[1] + n] = 1. - pos_dens[ch, n]
                    #O[i_org, 0, no_static_inputs + no_data_inputs + pos_dens.shape[1] + n] = 0.


def decide_trader(pos, alive, clk_sig, age, clk_counter, clk_lim, O, dec_thresh, max_CLK_period,
                  rr_counter, rr_lim, max_rr_period,
                  out_pos_idx, out_clk_lim_plus_idx, out_clk_lim_minus_idx,
                  out_rr_lim_plus_idx, out_rr_lim_minus_idx, out_rr_override,
                  out_thresh_plus_idx, out_thresh_minus_idx):
    no_org = O.shape[0]
    for i_org in range(no_org):
        if alive[i_org] == 1:
            current_thresh = dec_thresh[i_org]
            age[i_org] += 1
            clk_counter[i_org] += 1
            if clk_counter[i_org] > clk_lim[i_org]:
                clk_sig[i_org] = 1
                '''                if clk_sig[i_org] == 1:
                                    clk_sig[i_org] = 0
                                else:
                                    clk_sig[i_org] = 1'''
                clk_counter[i_org] = 0
            else:
                clk_sig[i_org] = 0

            reaction = 0
            reaction_override = 0
            # Check reaction override
            if O[i_org, 0, out_rr_override] > dec_thresh[i_org]:
                reaction_override = 1

            rr_counter[i_org] += 1
            if rr_counter[i_org] >= rr_lim[i_org] or reaction_override == 1:
                rr_counter[i_org] = 0
                reaction = 1

            # Decide buy/sell position

            # If all positions zero, keep same positions
            sum_zero = True
            for i_pos in range(pos.shape[1]):
                if O[i_org, 0, out_pos_idx + i_pos] > dec_thresh[i_org]:
                    sum_zero = False
            if not sum_zero and reaction:
                for i_pos in range(pos.shape[1]):
                    pos[i_org, i_pos] = 0.
                    if O[i_org, 0, out_pos_idx + i_pos] > dec_thresh[i_org]:
                        pos[i_org, i_pos] = 1.

            if O[i_org, 0, out_clk_lim_plus_idx] > dec_thresh[i_org]:  # increase clock period
                clk_lim[i_org] += 1
                if clk_lim[i_org] > max_CLK_period:
                    clk_lim[i_org] = max_CLK_period
            if O[i_org, 0, out_clk_lim_minus_idx] > dec_thresh[i_org]:  # decrease clock period
                clk_lim[i_org] -= 1
                if clk_lim[i_org] < 1:
                    clk_lim[i_org] = 1
            if O[i_org, 0, out_rr_lim_plus_idx] > dec_thresh[i_org]:  # increase reaction rate period
                rr_lim[i_org] += 1
                if rr_lim[i_org] > max_rr_period:
                    rr_lim[i_org] = max_rr_period
            if O[i_org, 0, out_rr_lim_minus_idx] > dec_thresh[i_org]:  # decrease reaction rate period
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


#  **************  Get Position Change Count  **************
@jit(nopython=True)
def position_change(pos, pos_last, pos_change):
    for i_org in range(pos.shape[0]):
        if i_org < pos.shape[0]:
            changed = 0.
            total_pos = 0.
            for i_pos in range(pos.shape[1]):
                total_pos += pos[i_org, i_pos] + pos_last[i_org, i_pos]
                if not pos[i_org, i_pos] == pos_last[i_org, i_pos]:
                    changed += 1.
                    break
            pos_change[i_org] += changed/total_pos


#  **************  Position Mask  **************
@jit(nopython=True)
def apply_position_mask(pos, pos_mask):
    """
    Given a position mask consisting of "1" (active) and "0" inactive, apply this mask to each organism's positions
    to deactivate positions which are no longer available.
    :param pos:
    :param pos_mask:
    :return:
    """
    for i_org in range(pos.shape[0]):
        pos_sum = 0
        if i_org < pos.shape[0]:
            for i_pos in range(pos.shape[1]):
                if pos_mask[i_pos] == 0:
                    pos[i_org, i_pos] = 0.
                else:
                    pos_sum += pos[i_org, i_pos]
        # Ensure new positions are not all 0
        if pos_sum == 0:
            pos[i_org, pos.shape[1]-1] = 1.

# **************  FITNESS  **************
def vector_fitness():
    a=1


def field_fitness():
    a=1


def roi_fitness2():
    a=1


def roi_fitness4():
    a=1

@jit(nopython=True)
def roi_fitness_loser_normalized(fitness, pos, data, t):
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
    # Sense state inputs
    for i_org in range(fitness.shape[0]):
        sum = 0.
        for i_pos in range(pos.shape[1]):
            sum += pos[i_org, i_pos]

        # normalize positions
        for i_pos in range(pos.shape[1]):
            pos[i_org, i_pos] = math.floor(pos[i_org, i_pos] / sum * 100) / 100

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
                roi = roi + pos[i_org, i_pos] / sum * data[i_pos, t] / data[i_pos, t+1]

        fitness[i_org] = fitness[i_org] * roi


@jit(nopython=True)
def roi_fitness(fitness, pos, pos_last, data, spread_ratio, t, fee_pct, use_spread, use_fee, loser_fit):
    """
    ROI fitness function for four currencies including the baseline (c_0=1). Include fees associated with the ask/bid
    spread rate.
    Fitness = current value = F_t = F_(t-1) * sum_i(c_i * p_i)
    :param fitness:
    :param pos:
    :param data: array of currency prices [N-1, T]
    :param spread ratio: the spread ratio (spread_percent/100). Array with length [N-1]
    :param t: current time index
    :return:
    """

    # Sense state inputs
    for i_org in range(fitness.shape[0]):
        sum = 0.
        for i_pos in range(pos.shape[1]):
            sum += pos[i_org, i_pos]

        sum_last = 0.
        for i_pos in range(pos.shape[1]):
            sum_last += pos_last[i_org, i_pos]

        if sum_last == 0:
            sum_last = 1

        # Calculate ROI
        if sum == 0:  # all positions 0. Set to USD
            roi = 1.
        else:
            #price_avg_current = 0.
            #price_avg_next = 0.
            roi = 0.
            roi_fee = 0.
            for i_pos in range(pos.shape[1]):
                roi = roi + pos[i_org, i_pos] / sum * data[i_pos, t + 1] / data[i_pos, t]
                #price_avg_current += data[i_pos, t] * pos[i_org, i_pos] / sum
                #price_avg_next += data[i_pos, t + 1] * pos[i_org, i_pos] / sum

                if use_spread == 2:  # Calculate spread based on average spread float
                    if not i_pos == (pos.shape[1]-1):  # ignore spread calculation on base currency
                        roi_fee = roi_fee
                elif use_spread == 1:  # Calculate spread based on detailed spread array
                    if not i_pos == (pos.shape[1] - 1):  # ignore spread calculation on base currency
                        sr = spread_ratio[i_pos * data.shape[1] + t]
                        if pos[i_org, i_pos] > pos_last[i_org, i_pos]:  # buy order spread fee
                            roi_fee = roi_fee - sr / (1 + sr) * math.fabs(pos[i_org, i_pos] / sum - pos_last[i_org, i_pos] / sum_last)
                        else:  # sell order spread fee
                            roi_fee = roi_fee - sr * math.fabs(pos_last[i_org, i_pos] / sum_last - pos[i_org, i_pos] / sum)
                else:
                    roi_fee = roi_fee
                if use_fee:  # Calculate fee
                    roi_fee = roi_fee - 0.5 * fee_pct[i_pos] / 100.0 * math.fabs(pos[i_org, i_pos] / sum - pos_last[i_org, i_pos] / sum_last)
            if loser_fit == 1:
                #roi = price_avg_current / price_avg_next + roi_fee
                roi = 1./roi - roi_fee
            else:
                #roi = price_avg_next / price_avg_current + roi_fee
                roi = roi + roi_fee

        fitness[i_org] = fitness[i_org] * roi


# ************** Reaction-Position Update ************
@jit(nopython=True)
def update_position(O, position_count, dec_thresh, input_pos_idx, out_pos_idx):

    # Sense state inputs
    for i_org in range(O.shape[0]):
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
def calc_pheromone_2d():
    a=1


# **************  Environment  **************
@jit(nopython=True)
def calc_numerical_closest_grid(grid, pos):
    # Find nearest neighbour value for each coordinate
    # Assign value to organism based on the field value of the nearest grid location

    no_grid_points, _ = grid.shape
    nearest_idx = -np.ones(pos.shape[0])
    for i_org in range(pos.shape[0]):
        dist_nearest = 999
        for ig in range(no_grid_points):
            dist = np.sqrt(np.sum(np.square(grid[ig, :] - pos[i_org, :])))
            if dist < dist_nearest:
                dist_nearest = dist
                nearest_idx[i_org] = ig

    return nearest_idx


@jit(nopython=True)
def calc_numerical_population_density_map(grid, pos, signal, field):

    no_grid_points, no_dims = grid.shape

    # calculate field based on organism values and grid-organism distances
    for ig in range(no_grid_points):  # for grid coordinate ig
        for i_org in range(pos.shape[0]):  # for organism i_org
            dist = np.sqrt(np.sum(np.square(grid[ig, :] - pos[i_org, :])))
            field[ig] += signal[i_org]/(dist + 0.3)

    # Find nearest neighbour value for each coordinate
    # Assign value to organism based on the field value of the nearest grid location
    org_values = -np.ones(pos.shape[0])
    for i_org in range(pos.shape[0]):
        dist_nearest = 999
        for ig in range(no_grid_points):
            dist = np.sqrt(np.sum(np.square(grid[ig, :] - pos[i_org, :])))
            if dist < dist_nearest:
                dist_nearest = dist
                org_values[i_org] = field[ig]

    return org_values

