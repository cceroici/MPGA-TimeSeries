import numpy as np
from numba import cuda
import math
import scipy.stats as stats
from src.gpu.array_gpu import _set_2d_array_zero, _set_1d_array_zero, _calc_multi_ch_array_2d_grad, _sum_atomic_channels, _normalize_array
from src.cpu.array_cpu import sum_atomic_channels_normalized
from src.gpu.genetic_sim_gpu import _find_closest_2d, _find_closest_2d_species, _calc_population_density, _calc_grid_field_ndim, _calc_apply_field_ndim
from src.cpu.genetic_sim_cpu import calc_numerical_population_density_map, calc_numerical_closest_grid


def calc_population_density(environment, population, sat_level=None, density_multiplier=1, PD_channel=0):
    threadsperblock = (8, 8)
    blockspergrid_x = math.ceil(environment.sim_size[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(environment.sim_size[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    if sat_level is None:
        sat_level = 1E8
    _set_2d_array_zero[blockspergrid, threadsperblock](environment.d_PD, PD_channel)  # Reset pheromone array to zeros
    _calc_population_density[blockspergrid, threadsperblock](environment.d_PD, population.d_pos,
                                                             environment.sim_size[0], environment.sim_size[1],
                                                             sat_level, density_multiplier, PD_channel)
    _calc_multi_ch_array_2d_grad[blockspergrid, threadsperblock](environment.d_PD,
                                                                 environment.d_PD_dx, environment.d_PD_dy,
                                                                 PD_channel)


def calc_position_density(environment, population, channel, USE_GPU=True):
    no_pos = environment.posD.shape[1]

    if USE_GPU:

        # Zero position density array
        threadsperblock = 16
        blockspergrid = math.ceil(no_pos / threadsperblock)
        _set_1d_array_zero[blockspergrid, threadsperblock](environment.d_posD, channel)

        # Calculate position density
        threadsperblock = 128
        blockspergrid = math.ceil(population.pop_count / threadsperblock)
        for ip in range(no_pos):
            _sum_atomic_channels[blockspergrid, threadsperblock](environment.d_posD, population.d_pos, ip, channel)
        _normalize_array[1, 8](environment.d_posD, channel)
    else:
        for ip in range(no_pos):
            environment.posD[channel, ip] = sum_atomic_channels_normalized(population.pos, ip, channel)
        environment.posD[channel, :] = environment.posD[channel, :]/environment.posD[channel, :].sum()


def calc_numerical_field(population, env, USE_GPU=True, field_type='pop-density'):

    if field_type == 'pop-density':
        if USE_GPU:
            signal = cuda.to_device(np.ones(population.pop_count))
        else:
            signal = np.ones(population.pop_count)
    elif field_type == 'pheromone':
        if USE_GPU:
            signal = env.d_pher
        else:
            signal = env.pher

    if USE_GPU:
        threadsperblock = 128
        blockspergrid = math.ceil(env.numerical_grid_coords.shape[0] / threadsperblock)

        d_field = cuda.to_device(np.zeros(env.numerical_grid_coords.shape[0]))
        d_signal = cuda.to_device(signal)
        _calc_grid_field_ndim[blockspergrid, threadsperblock](env.d_numerical_grid_coords, population.d_pos, d_signal, d_field)

        threadsperblock = 128
        blockspergrid = math.ceil(population.d_pos.shape[0] / threadsperblock)
        _calc_apply_field_ndim[blockspergrid, threadsperblock](env.d_numerical_grid_coords, population.d_pos, d_field, env.d_PD)
    else:
        field_map = np.zeros(env.numerical_grid_coords.shape[0])
        env.PD[population.PD_channel, :, :] = calc_numerical_population_density_map(grid=env.numerical_grid_coords, pos=population.d_pos,
                                                       signal=signal, field=field_map)
        #values = values / values.max()


def initial_positions(no_org, sigma, sim_size=None, position_count=None, distribution_type='Gaussian'):
    """

    :param no_org: Number of organisms in population
    :param sigma: For gaussian distributions, the variance
    :param sim_size: For '2D' simulation, the coordinate space boundaries
    :param position_count: For 'numerical' simulation, the number of different possible positions
    :param distribution_type:
    :return:
    """

    # 2D Simulation
    rand_pos = np.zeros((no_org, position_count))
    if sim_size is not None:
        if distribution_type == 'Gaussian':
            rand_pos = np.random.randn(no_org, 2) * sigma - sigma / 2
        elif distribution_type == 'uniform':
            rand_pos = np.random.rand(no_org, 2)
            rand_pos[:, 0] = rand_pos[:, 0] * sim_size[0] - sim_size[0] / 2
            rand_pos[:, 1] = rand_pos[:, 1] * sim_size[1] - sim_size[1] / 2
        else:
            rand_pos = None
        rand_pos[rand_pos > sim_size[0] / 2] = sim_size[0] / 2 - 1
        rand_pos[rand_pos > sim_size[1] / 2] = sim_size[1] / 2 - 1
        rand_pos[rand_pos < -sim_size[0] / 2] = -sim_size[0] / 2 + 1
        rand_pos[rand_pos < -sim_size[1] / 2] = -sim_size[1] / 2 + 1
    # numerical simulation
    elif position_count is not None:
        if distribution_type == 'numerical':
            rand_pos = np.random.rand(no_org, position_count)
            rand_pos = np.divide(rand_pos, np.transpose(np.tile(np.sum(rand_pos, 1), [position_count, 1]), (1,0)))
        elif distribution_type == 'numerical-binary':
            rand_pos = np.round(np.random.rand(no_org, position_count))
            #print("USING ZERO INITIAL POS")
            #rand_pos = np.zeros(rand_pos.shape)
            # if organism has all 0 positions, assume all funds assigned to usd (last position index)
            sum = np.sum(rand_pos, 1)
            sum_zero = sum==0
            null_pos = np.zeros(position_count)
            null_pos[position_count-1] = 1.
            rand_pos[sum_zero, :] = null_pos
        elif distribution_type == 'zero':
            rand_pos = np.zeros((no_org, position_count))

    return rand_pos


def get_truncated_normal(mean=0, sd=None, low=0, upp=10, p_start = 0.05):
    if sd is None:
        z0 = stats.norm.ppf(p_start)
        sd = -mean/z0
    return stats.truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class SimEnvironment():
    def __init__(self, sim_size, sim_type, time_steps, pher_no_channels, PD_no_channels, numerical_position_count=None,
                 game_config=None):
        super(SimEnvironment, self).__init__()

        self.sim_size = sim_size
        self.time_steps = time_steps
        self.sim_type = sim_type
        self.numerical_position_count = numerical_position_count
        self.spread_mode = "none"
        self.game_config = game_config

        # field arrays
        self.field = None
        self.field_dx = None
        self.field_dy = None
        self.input_data = None
        self.cost_metric = None

        # numerical data arrays
        self.data_in = None
        self.data_out = None
        self.data_in_names = None
        self.data_dates = None
        self.position_density_names = None
        self.position_mask = None  # Array mask for toggling specific output positions. '1' for active, '0' for inactive
        self.numerical_grid_coords = None
        self.numerical_grid_nearest_org = None
        self.numerical_position_labels = None
        self.env_data_file = None
        self.validation_start_sample = None
        self.spread_ratio_avg = None  # scalar average spread ratio for each position
        self.spread_ratio = None  # details sampled spread ratio for each position
        self.trading_fees = None

        # pheromone arrays
        self.pher_no_channels = pher_no_channels
        self.pher = None
        self.pher_dx = None
        self.pher_dy = None

        # population density arrays
        self.PD_no_channels = PD_no_channels
        self.PD = None
        self.PD_dx = None
        self.PD_dy = None

        # position density arrays
        self.PosD_no_channels = PD_no_channels
        self.PosD = None

        # GPU field arrays
        self.d_field = None
        self.d_field_dx = None
        self.d_field_dy = None
        self.d_input_data = None
        self.d_cost_metric = None
        self.d_position_map = None
        self.d_spread_ratio_avg = None
        self.d_spread_ratio = None

        # GPU numerical arrays
        self.d_data_in = None
        self.d_data_out = None
        self.d_numerical_grid_coords = None
        self.d_numerical_grid_nearest_org = None
        self.d_trading_fees = None
        self.d_position_mask = None

        # GPU pheromone arrays
        self.d_pher = None
        self.d_pher_dx = None
        self.d_pher_dy = None

        # GPU population density arrays
        self.d_PD = None
        self.d_PD_dx = None
        self.d_PD_dy = None

        # GPU position density arrays
        self.d_PosD = None

    def field_calc_grad(self):
        self.field_dx, self.field_dy, _ = np.gradient(self.field)
        self.field_dx *= self.sim_size[0]
        self.field_dy *= self.sim_size[1]

    def field_cpu2gpu(self):
        self.d_field = cuda.to_device(self.field)
        self.d_field_dx = cuda.to_device(self.field_dx)
        self.d_field_dy = cuda.to_device(self.field_dy)
        if self.input_data is not None:
            self.d_input_data = cuda.to_device(self.input_data)
        if self.cost_metric is not None:
            self.d_cost_metric = cuda.to_device(self.cost_metric)

    def pheromone_cpu2gpu(self):
        self.d_pher = cuda.device_array((self.pher_no_channels, self.sim_size[0], self.sim_size[1]), dtype='float',
                                        strides=None, order='C', stream=0)
        self.d_pher_dx = cuda.device_array((self.pher_no_channels, self.sim_size[0], self.sim_size[1]), dtype='float',
                                           strides=None, order='C', stream=0)
        self.d_pher_dy = cuda.device_array((self.pher_no_channels, self.sim_size[0], self.sim_size[1]), dtype='float',
                                           strides=None, order='C', stream=0)

        self.pher = np.zeros(self.d_pher.shape)
        self.pher_dx = np.zeros(self.d_pher.shape)
        self.pher_dy = np.zeros(self.d_pher.shape)

    def population_density_cpu2gpu(self, GRAD=True):
        if self.sim_type=="2D":
            self.d_PD = cuda.device_array((self.PD_no_channels, self.sim_size[0], self.sim_size[1]), dtype='float',
                                          strides=None, order='C', stream=0)
            self.PD = np.zeros(self.d_PD.shape)

            if GRAD:
                self.d_PD_dx = cuda.device_array((self.PD_no_channels, self.sim_size[0], self.sim_size[1]), dtype='float',
                                                 strides=None, order='C', stream=0)
                self.d_PD_dy = cuda.device_array((self.PD_no_channels, self.sim_size[0], self.sim_size[1]), dtype='float',
                                                 strides=None, order='C', stream=0)

                self.PD_dx = np.zeros(self.d_PD.shape)
                self.PD_dy = np.zeros(self.d_PD.shape)

    def position_density_cpu2gpu(self, GPU=True):
        """
        Initialize/allocate memory for position density (numerical simulation)
        The position density arrays have shape: [No. position density channels, No. positions]
        This represents the density of the '1' state
        This only works for binary states ('1' or '0'), since we can simply do 1-posD to get the '0' state
        :return:
        """

        self.posD = np.zeros((self.PosD_no_channels, self.numerical_position_count))
        if GPU:
            self.d_posD = cuda.to_device(self.posD)


    def load_numerical_data(self, data_type, populations=None, GPU_arrays=True, duration=None, is_validation=False,
                            truncated_normal_distr_start=True, BLIND_TRAIN=False, start_t_force=None,
                            environment_parameters=None):
        if data_type == "numerical-data-file":
            if self.env_data_file is None:
                import pickle
                self.env_data_file = pickle.load(open(environment_parameters['data file'], "rb"))
                self.validation_start_sample = None
                if "validation date" in list(self.env_data_file.keys()) and duration is not None:
                    if self.env_data_file["validation date"] is not None and self.env_data_file[
                        "validation idx"] is not None:
                        self.validation_start_sample = self.env_data_file["validation idx"]
                        print("******* Validation Strategy: Using preset validation start sample {} (date: {}) with {} validation samples".format(
                            self.env_data_file["validation idx"], self.env_data_file["validation date"], self.env_data_file["input data"].shape[1] - self.validation_start_sample))
                    if (self.env_data_file["input data"].shape[1] - self.validation_start_sample) < duration:
                        print(
                            "WARNING (Environment.load_numerical_data): Preset validation start sample ({}) leaves only {} samples for "
                            "validation, whereas the desired generation duration is {}. Switching to automated validation sample".format(
                                self.env_data_file["input data"].shape[1],
                                self.env_data_file["input data"].shape[1] - self.validation_start_sample,
                                duration))
                        self.validation_start_sample = None
                if self.validation_start_sample is None and duration is not None:
                    input_data = self.env_data_file["input data"]
                    #self.validation_start_sample = int(
                    #    np.ceil(input_data.shape[1] * (1. - validation_proportion - 0.02)))
                    self.validation_start_sample = input_data.shape[1] - duration - 10
                    print("******* Validation Strategy: Using automatic validation start sample {} / {} ({} validation samples)".format(
                        self.validation_start_sample, input_data.shape[1], input_data.shape[1]-self.validation_start_sample))
                    # Check that the validation start sample allows for at least "duration" samples in validation
                    if (self.env_data_file["input data"].shape[1] - self.validation_start_sample) < duration:
                        print(
                            "ERROR (Environment.load_numerical_data): Validation start sample ({}) leaves only {} samples for "
                            "validation, whereas the desired generation duration is {}".format(
                                self.env_data_file["input data"].shape[1],
                                self.env_data_file["input data"].shape[1] - self.validation_start_sample,
                                duration))
                        return
                if BLIND_TRAIN:
                    print("******* Using \"Blind\" training (training data includes validation period)")

                # Check for and load spread data
                any_use_spread = False
                for pop in populations:
                    if pop.use_spread:
                        any_use_spread = True
                        break
                if any_use_spread:
                    if "spread rate" in list(self.env_data_file.keys()):
                        spread_data = self.env_data_file["spread rate"]
                        if spread_data is not None:
                            self.spread_ratio = np.array(spread_data)
                            print("Using detailed spread data with {} samples for {} position pairs".format(
                                self.spread_ratio.shape[1], self.spread_ratio.shape[0]))
                            if GPU_arrays:
                                self.d_spread_ratio = cuda.to_device(self.spread_ratio)
                            self.spread_mode = "detailed"

                    elif "spread pct" in list(self.env_data_file.keys()):
                        spread_data = self.env_data_file["spread pct"]
                        if spread_data is not None:
                            self.spread_ratio_avg = np.array(spread_data) / 100
                            print("Using average spread data for {} position pairs".format(len(self.spread_ratio_avg)))
                            if GPU_arrays:
                                self.d_spread_ratio_avg = cuda.to_device(self.spread_ratio_avg)
                            self.spread_mode = "average"
                #print("SETTING SPREAD INPUT DATA TO 0")
                if "position mask" in self.env_data_file:
                    self.position_mask = self.env_data_file["position mask"]
                    print("Masking/disabling output positions: {}".format(
                        list(np.array(self.env_data_file['currencies'][:-1])[self.position_mask == 0.])))

            input_data = self.env_data_file["input data"]
            price_data = self.env_data_file["price data"]
            currencies = self.env_data_file["currencies"]
            self.trading_fees = self.env_data_file["trade fee"]  # transaction fee array (for each currency) in percent
            dates = self.env_data_file['dates']


            # Data input labels (anything not specifically related to the organism)
            # Things like data/technical inputs, density, and pheromone inputs are stored in the environment
            data_input_names = self.env_data_file["input data labels"]

            start_sample = 0

            if is_validation:
                if start_t_force is not None:  # Specified start time
                    start_t = start_t_force
                    if duration is None or (start_t + duration) > len(self.env_data_file['dates']):
                        duration = len(self.env_data_file['dates']) - start_t - 1
                elif duration is None:  # For full dataset validation
                    start_t = 0
                    duration = self.env_data_file["input data"].shape[1]-15
                    if "validation idx" in list(self.env_data_file.keys()):
                        if self.env_data_file["validation idx"] is not None:
                            self.validation_start_sample = self.env_data_file["validation idx"]
                elif duration == -1:
                    # If T=-1, use the full remaining validation data samples
                    duration = len(self.env_data_file['dates']) - self.validation_start_sample - 1
                    print("Using full remaining data for validation run ({} samples)".format(duration))
                    start_t = self.validation_start_sample
                else:
                    start_t = self.validation_start_sample
            else:
                if BLIND_TRAIN:
                    last_training_sample = self.env_data_file["input data"].shape[1] - duration - 10
                else:
                    last_training_sample = self.validation_start_sample - duration
                if truncated_normal_distr_start:
                    X = get_truncated_normal(mean=last_training_sample, p_start=0.02, low=0,
                                             upp=last_training_sample)
                    start_t = int(X.rvs())
                else:
                    start_t = int(np.random.randint(start_sample, last_training_sample))

            position_density_names = []
            if populations is None:
                for currency in currencies:
                    position_density_names.append("PosD " + currency + "1")
                for currency in currencies:
                    position_density_names.append("PosD " + currency + "0")
            else:
                for p in range(len(populations)):
                    pos_dens_name_prefix = ""
                    if len(populations) > 1:
                        pos_dens_name_prefix = "{}{}-".format(p, populations[p].pop_label)
                    for currency in currencies:
                        position_density_names.append(pos_dens_name_prefix + "PosD " + currency + "1")
                for p in range(len(populations)):
                    pos_dens_name_prefix = ""
                    if len(populations) > 1:
                        pos_dens_name_prefix = "{}{}-".format(p, populations[p].pop_label)
                    for currency in currencies:
                        position_density_names.append(pos_dens_name_prefix + "PosD " + currency + "0")

            self.data_in = np.ascontiguousarray(input_data[:, start_t:(start_t + duration + 10)])
            self.data_out = np.ascontiguousarray(price_data[:, start_t:(start_t + duration + 10)])
            self.data_dates = dates[start_t:(start_t + duration + 10)]
            if self.data_in.shape[1] < duration:
                print("ERROR (Environment.load_numerical_data): Data input length ({}) is less than the specified "
                      "generation simulation duration ({}). Is validation: {}".format(self.data_in.shape[1], duration,
                                                                                      is_validation))
                self.data_in = None
                self.data_out = None
                return

            # self.data_out = np.ascontiguousarray(price_data[:, (start_t-1):(start_t+duration+10)])  # *********** SANITY CHECK
            self.data_in_names = data_input_names
            self.position_density_names = position_density_names
            self.numerical_position_labels = currencies
        elif data_type == "fixed-dataset":
            pass
        else:
            print("ERROR (Environment.py:load_numerical_data()): Unknown data_type '{}'".format(data_type))

        if GPU_arrays:
            self.d_data_in = cuda.to_device(self.data_in)
            self.d_data_out = cuda.to_device(self.data_out)
            if self.trading_fees is not None:
                self.d_trading_fees = cuda.to_device(self.trading_fees)
            if self.position_mask is not None:
                self.d_position_mask = cuda.to_device(self.position_mask)

    def initialize_game_env(self, game_config, populations, GPU_arrays=True):

        # data in represents the current game state
        self.data_in = np.ascontiguousarray(game_config.initial_game_state_array())

        position_density_names = []
        if populations is None:
            for p in game_config.decision_names:
                position_density_names.append("PosD_" + p + "_1")
            for p in game_config.decision_names:
                position_density_names.append("PosD_" + p + "_0")
        else:
            for p in range(len(populations)):
                pos_dens_name_prefix = ""
                if len(populations) > 1:
                    pos_dens_name_prefix = "{}{}-".format(p, populations[p].pop_label)
                for p in game_config.decision_names:
                    position_density_names.append(pos_dens_name_prefix + "PosD_" + p + "_1")
                for p in game_config.decision_names:
                    position_density_names.append(pos_dens_name_prefix + "PosD_" + p + "_0")

        # self.data_out = np.ascontiguousarray(price_data[:, (start_t-1):(start_t+duration+10)])  # *********** SANITY CHECK
        self.data_in_names = game_config.input_names
        self.position_density_names = position_density_names
        self.numerical_position_labels = game_config.decision_names

        if GPU_arrays:
            self.d_data_in = cuda.to_device(self.data_in)

    def generate_numerical_grid(self, no_dims, no_steps, grid_lims=(0, 1), USE_GPU=True):
        """
        Generate a list of grid coordinates for N-dimensions
        Format: [no_points, no_dims] = [no_steps^no_dims, no_dims]
        :param no_dims:
        :param no_steps:
        :param grid_lims:
        :return:
        """
        y = []
        for id in range(no_dims):
            y.append(np.linspace(grid_lims[0], grid_lims[1], no_steps))
        meshgrid = np.array(np.meshgrid(*y))
        self.numerical_grid_coords = np.transpose(meshgrid.reshape(meshgrid.shape[0], no_steps ** no_dims), (1, 0))

        self.PD = np.zeros((self.PD_no_channels, self.numerical_grid_coords.shape[0]))

        if USE_GPU:
            self.d_numerical_grid_coords = cuda.to_device(self.numerical_grid_coords)
            self.d_PD = cuda.device_array((self.PD_no_channels, self.numerical_grid_coords.shape[0]), dtype='float',
                                          strides=None, order='C', stream=0)


