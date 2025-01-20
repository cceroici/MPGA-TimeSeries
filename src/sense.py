import numpy as np
import math
from src.gpu.genetic_sim_gpu import _sense_trader, _sense_gamer, _update_position
from src.cpu.genetic_sim_cpu import sense_trader, update_position


def calc_sense_numerical(population, env, t_step, USE_GPU=True):

    # Find input node indexes
    input_start_idx = 0
    input_pos_idx = input_start_idx
    input_clk_idx = input_pos_idx + population.pos.shape[1]
    input_age_idx = input_pos_idx + population.pos.shape[1] + 1

    input_data_idx = input_pos_idx + population.pos.shape[1] + 2

    #no_pos_density_channels = env.posD.shape[0]
    # input pos_density_idx =              static_inputs + 2            + data_inputs
    input_pos_density_idx = input_pos_idx + population.pos.shape[1] + 2 + population.template.data_input_count

    # Update the last position array
    population.update_pos_last(USE_GPU=USE_GPU)

    if USE_GPU:
        if population.pop_count < 2000:
            threadsperblock = 8
        elif population.pop_count < 4000:
            threadsperblock = 16
        else:
            threadsperblock = 32
        blockspergrid = math.ceil(population.pop_count / threadsperblock)

        _sense_trader[blockspergrid, threadsperblock](population.d_O, population.d_pos, population.d_age,
                                                      population.d_alive, population.d_clk_sig,
                                                      env.d_data_in,
                                                      env.d_posD,
                                                      t_step,
                                                      input_pos_idx, input_clk_idx, input_age_idx, input_data_idx,
                                                      input_pos_density_idx)

        # Overwrite the position input according to the existing outputs (to update the sense inputs for any organisms which haven't triggered reactions)
        _update_position[blockspergrid, threadsperblock](population.d_O, population.template.position_count, population.decision_threshold, input_pos_idx, population.out_offset)
    else:
        sense_trader(pos=population.pos, alive=population.alive, clk_sig=population.clk_sig, age=population.age,
                     data=env.data_in, pos_dens=env.posD, pos_dens_ch=population.PD_channel, O=population.O, t=t_step,
                     input_pos_idx=input_pos_idx, input_clk_idx=input_clk_idx, input_age_idx=input_age_idx,
                     input_data_idx=input_data_idx, input_pos_density_idx=input_pos_density_idx)

        # Overwrite the position input according to the existing outputs (to update the sense inputs for any organisms which haven't triggered reactions)
        update_position(population.O, population.template.position_count, population.decision_threshold, input_pos_idx, population.out_offset)


def calc_sense_game(population, env, USE_GPU=True):

    # Find input node indexes
    input_start_idx = 0
    input_pos_idx = input_start_idx
    input_actb_idx = population.template.in_idx["actb"]
    input_data_idx = population.template.data_input_start_idx
    input_clk_idx = -1
    input_age_idx = -1

    no_pos_density_channels = env.posD.shape[0]
    input_pos_density_idx = population.template.position_density_start_idx

    if USE_GPU:
        if population.pop_count < 2000:
            threadsperblock = 8
        elif population.pop_count < 4000:
            threadsperblock = 16
        else:
            threadsperblock = 32
        blockspergrid = math.ceil(population.pop_count / threadsperblock)

        _sense_gamer[blockspergrid, threadsperblock](population.d_O, population.d_pos, population.d_age,
                                                     population.d_alive, population.d_clk_sig, population.d_actb_score,
                                                     env.d_data_in,
                                                     env.d_posD,
                                                     input_pos_idx, input_actb_idx,
                                                     input_data_idx,
                                                     input_pos_density_idx)

        # Overwrite the position input according to the existing outputs (to update the sense inputs for any organisms which haven't triggered reactions)
        _update_position[blockspergrid, threadsperblock](population.d_O, population.template.game_decision_count, population.decision_threshold, input_pos_idx, population.out_offset)


def calc_sense(population, env, t_step, sim_type, USE_GPU=True):

    if sim_type == "numerical":
        calc_sense_numerical(population=population, env=env, t_step=t_step, USE_GPU=USE_GPU)
    elif sim_type == "game":
        calc_sense_game(population=population, env=env, USE_GPU=USE_GPU)

    if population.debug_target_org is not None:
        population.debug_pos = np.zeros(population.d_pos.shape)
        population.d_pos.copy_to_host(population.debug_pos)
        population.debug_input_states = np.zeros(population.O.shape)
        population.d_O.copy_to_host(population.debug_input_states)
        population.debug_input_states = population.debug_input_states[population.debug_target_org, 0, :]
        population.debug_field = env.field[:, :, t_step]
