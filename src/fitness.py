import numpy as np
import math
from numba import cuda
from src.gpu.genetic_sim_gpu import _vector_fitness, _field_fitness, _roi_fitness, _roi_fitness_stabilizer_normalized, _roi_fitness_destabilizer_normalized
from src.cpu.genetic_sim_cpu import *


def vector_fitness(d_vec, d_fitness):
    if d_vec.shape[0] < 2000:
        threadsperblock = 8
    elif d_vec.shape[0] < 4000:
        threadsperblock = 16
    else:
        threadsperblock = 32
    blockspergrid = math.ceil(d_vec.shape[0] / threadsperblock)
    _vector_fitness[blockspergrid, threadsperblock](d_fitness, d_vec)


def field_fitness(d_fitness, d_pos, d_field, t, sim_width, sim_height):
    if d_fitness.shape[0] < 2000:
        threadsperblock = 8
    elif d_fitness.shape[0] < 4000:
        threadsperblock = 16
    else:
        threadsperblock = 32
    blockspergrid = math.ceil(d_fitness.shape[0] / threadsperblock)
    _field_fitness[blockspergrid, threadsperblock](d_fitness, d_pos, d_field, t, sim_width, sim_height)


def roi_numerical(fitness, pos, pos_last, data, t, spread_ratio, fee_pct, use_spread=0, use_fee=0, USE_GPU=True,
                  loser=False):
    if loser:
        use_fee = 0
        use_spread = 0
    if USE_GPU:
        if fitness.shape[0] < 2000:
            threadsperblock = 8
        elif fitness.shape[0] < 4000:
            threadsperblock = 16
        else:
            threadsperblock = 32
        blockspergrid = math.ceil(fitness.shape[0] / threadsperblock)

        if use_spread == 0:
            d_spread_ratio = cuda.device_array((5, 5), dtype='float', strides=None, order='C', stream=0)
        else:
            d_spread_ratio = spread_ratio

        _roi_fitness[blockspergrid, threadsperblock](fitness,
                                                     pos, pos_last,
                                                     data,
                                                     d_spread_ratio,
                                                     t,
                                                     fee_pct,
                                                     use_spread,
                                                     use_fee, 1 if loser else 0)
    else:
        roi_fitness(fitness=fitness, pos=np.array(pos).astype('float32'), pos_last=np.array(pos_last).astype('float32'),
                    data=data, spread_ratio=np.array(spread_ratio).astype('float32').flatten(), t=t, fee_pct=np.array(fee_pct),
                    use_spread=use_spread, use_fee=use_fee, loser_fit=1 if loser else 0)


def roi_numerical_stabilizer(d_fitness, d_pos, d_data, t, sensitivity=1):
    if d_fitness.shape[0] < 2000:
        threadsperblock = 8
    elif d_fitness.shape[0] < 4000:
        threadsperblock = 16
    else:
        threadsperblock = 32
    blockspergrid = math.ceil(d_fitness.shape[0] / threadsperblock)
    '''
        pos = np.zeros(d_pos.shape)
        fitness = np.zeros(d_fitness.shape)
        data = np.zeros(d_data.shape)
        d_pos.copy_to_host(pos)
        d_fitness.copy_to_host(fitness)
        d_data.copy_to_host(data)
        # print(pos[0, :])
        # print("data gain: {}, fitness[0]={}".format(data[0,t+1]/data[0,t], fitness[0]))
        track_org = 0
        if t < (data.shape[1]-1):
            print("t={}, org {}: fitness={:0.3f}, pos={}, gain: [{}*{:0.2f}, {}*{:0.2f}, {}*{:0.2f}, {}*{:0.2f}]".format(t, track_org, fitness[track_org], pos[track_org],
                                                                                                pos[track_org][0], data[0, t + 1] / data[0, t],
                                                                                                pos[track_org][1],
                                                                                                data[1, t + 1] / data[1, t],
                                                                                                pos[track_org][2],
                                                                                                data[2, t + 1] / data[2, t],
                                                                                                pos[track_org][3],
                                                                                                1))'''
    _roi_fitness_stabilizer_normalized[blockspergrid, threadsperblock](d_fitness, d_pos, d_data, t, sensitivity)


def roi_numerical_destabilizer(d_fitness, d_pos, d_data, t, sensitivity=1.0, offset=0.99):
    if d_fitness.shape[0] < 2000:
        threadsperblock = 8
    elif d_fitness.shape[0] < 4000:
        threadsperblock = 16
    else:
        threadsperblock = 32
    blockspergrid = math.ceil(d_fitness.shape[0] / threadsperblock)
    '''
        pos = np.zeros(d_pos.shape)
        fitness = np.zeros(d_fitness.shape)
        data = np.zeros(d_data.shape)
        d_pos.copy_to_host(pos)
        d_fitness.copy_to_host(fitness)
        d_data.copy_to_host(data)
        # print(pos[0, :])
        # print("data gain: {}, fitness[0]={}".format(data[0,t+1]/data[0,t], fitness[0]))
        track_org = 0
        if t < (data.shape[1]-1):
            print("t={}, org {}: fitness={:0.3f}, pos={}, gain: [{}*{:0.2f}, {}*{:0.2f}, {}*{:0.2f}, {}*{:0.2f}]".format(t, track_org, fitness[track_org], pos[track_org],
                                                                                                pos[track_org][0], data[0, t + 1] / data[0, t],
                                                                                                pos[track_org][1],
                                                                                                data[1, t + 1] / data[1, t],
                                                                                                pos[track_org][2],
                                                                                                data[2, t + 1] / data[2, t],
                                                                                                pos[track_org][3],
                                                                                                1))'''
    _roi_fitness_destabilizer_normalized[blockspergrid, threadsperblock](d_fitness, d_pos, d_data, t, sensitivity,
                                                                         offset)

def calc_fitness_roi_validation(data_charts, positions, fee_pct=None, spread_ratio=None, debug=False):
    T, no_pos = positions.shape

    fitness = []
    fitness_current = 1.
    if debug: print("---------------------**************")

    pos_last = np.copy(positions[0, :])
    for t in range(T):
        #price_avg_current = 0
        #price_avg_next = 0
        roi = 0.
        roi_fee = 0
        if t > 0:
            pos_last = positions[t-1, :]
        for i_pos in range(no_pos):
            roi += positions[t, i_pos] * data_charts[i_pos, t + 1]/data_charts[i_pos, t]
            #price_avg_current += data_charts[i_pos, t] * positions[t, i_pos]
            #price_avg_next += data_charts[i_pos, t + 1] * positions[t, i_pos]

            if fee_pct is not None:
                if fee_pct[i_pos] > .0:
                    roi_fee -= 0.5*fee_pct[i_pos]/100*np.abs(positions[t, i_pos] - pos_last[i_pos])

            if spread_ratio is not None and not i_pos == (no_pos - 1):
                pos_sum = np.sum(positions[t, :])
                pos_last_sum = np.sum(pos_last)
                if len(spread_ratio.shape) == 2:  # detailed spread data
                    if positions[t, i_pos] > pos_last[i_pos]:  # buy order spread
                        roi_fee -= spread_ratio[i_pos, t]/(1+spread_ratio[i_pos, t]) *\
                              np.abs(positions[t, i_pos]/pos_sum - pos_last[i_pos]/pos_last_sum)
                    else:  # sell order spread fee
                        roi_fee -= spread_ratio[i_pos, t] * np.abs(pos_last[i_pos]/pos_last_sum - positions[t, i_pos]/pos_sum)
                elif len(spread_ratio.shape) == 1:  # average spread
                    print("ERROR: average spread not implemented in \"Calc_Fitness_ROI_Validation()\" ")
                    pass
        #roi = price_avg_next/price_avg_current - roi_fee
        roi += roi_fee

        fitness_current *= roi
        fitness.append(fitness_current)

        if debug: print(
            "t={}, fitness={:.3f}, roi={:.3f}, pos={}, {}data={}".format(t, fitness_current, roi, positions[t, :],
                                                                 "roi fee: {:.3f}, pos_last={}, ".format(roi_fee, pos_last) if fee_pct > 0. else "",
                                                                 data_charts[:, t]))

    return np.array(fitness)


def calc_fitness(fitness_type, population, t, env=None, d_vec=None, d_price=None, d_field=None, N_roi_currencies=2,
                 d_pos_map=None, USE_GPU=True):
    if fitness_type == "ROI-numerical" or fitness_type == "ROI-numerical-loser":  # Dimensionless data inputs representing financial data
        spread = None
        spread_mode = 0
        if population.use_spread:
            spread_mode = 1 if env.spread_mode == "detailed" else 2
            if USE_GPU:
                spread = env.d_spread_ratio if env.spread_mode == "detailed" else env.d_spread_ratio_avg
            else:
                spread = env.spread_ratio if env.spread_mode == "detailed" else env.spread_ratio_avg

        roi_numerical(fitness=population.d_fitness if USE_GPU else population.fitness,
                      pos=population.d_pos if USE_GPU else population.pos,
                      pos_last=population.d_pos_last if USE_GPU else population.pos_last,
                      data=env.d_data_out if USE_GPU else env.data_out,
                      t=t,
                      spread_ratio=spread,
                      use_spread=spread_mode,
                      fee_pct=env.d_trading_fees if USE_GPU else env.trading_fees,
                      use_fee=1 if env.trading_fees is not None and population.use_trans_fee else 0,
                      USE_GPU=USE_GPU, loser=fitness_type == "ROI-numerical-loser"
                      )
    elif USE_GPU:
        if fitness_type == "field-health":
            field_fitness(d_fitness=population.d_fitness, d_field=d_field, d_pos=population.d_pos,
                          t=t, sim_width=env.sim_size[0],
                          sim_height=env.sim_size[1])
        elif fitness_type == "ROI":  # The 2D spatial implementation of fitness, where the environment represents input data
            roi_fitness(d_fitness=population.d_fitness, d_pos=population.d_pos, d_P=d_price, t=t,
                        sim_width=env.sim_size[0], sim_height=env.sim_size[1], N=N_roi_currencies,
                        d_pos_map=d_pos_map)
        elif fitness_type == "ROI-numerical-stabilizer":  # Dimensionless data inputs representing financial data. Aiming for ROI=1
            roi_numerical_stabilizer(d_fitness=population.d_fitness, d_pos=population.d_pos, d_data=env.d_data_out,
                                     t=t)
        elif fitness_type == "ROI-numerical-destabilizer":  # Dimensionless data inputs representing financial data. Aiming for ROI=1
            roi_numerical_destabilizer(d_fitness=population.d_fitness, d_pos=population.d_pos,
                                       d_data=env.d_data_out, t=t)
        elif fitness_type == "vector":
            vector_fitness(d_vec=d_vec, d_fitness=population.d_fitness)
        elif fitness_type == "vector-health":
            vector_fitness(d_vec=population.d_health, d_fitness=population.d_fitness)
        else:
            print("WARNING (Calc_Fitness): Invalid fitness calculation type: {}".format(fitness_type))
    else:  # CPU
        if fitness_type == "ROI-numerical":  # Dimensionless data inputs representing financial data
            roi_numerical(fitness=population.fitness, pos=population.pos, data=env.data_out, t=t, USE_GPU=False)
        elif fitness_type == "ROI-numerical-loser":  # Dimensionless data inputs representing financial data
            roi_numerical(fitness=population.fitness, pos=population.pos, data=env.data_out, t=t, USE_GPU=False)
        else:
            print(
                "************ ERROR!!!!!!! ******** Fitness.py:Calc_Fitness() {} has no CPU implementation".format(
                    fitness_type))


if __name__ == "__main__":
    age = np.random.rand(1000)
    fitness = np.empty(age.shape[0])
