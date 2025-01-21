import numpy as np
import json
from tqdm import tqdm
from datetime import datetime
from src.species import Population, OrganismDebugProbe
from src.visualize import Visualizer, graph_preview, numerical_visualization_set
from src.environment import SimEnvironment, calc_position_density
from src.repopulation import tournament_repopulate
from src.genome import genetic_diversity
from src.sense import calc_sense
from src.fitness import calc_fitness
from src.profiling import GsimProfiler
from src.position_change import update_position_change
from tensorboardX import SummaryWriter
from simulations.games.TicTacToe_utils import TicTacToe_animation
from simulations.games.TicTacToe_utils import legal_move, update_game_state, check_winner

"""
Genetic Simulation Module/Class definition
"""


class GeneticSim:
    def __init__(self, species_templates, species_configurations, time_steps_per_gen, path_config,
                 genetic_sim_params, environment_parameters,
                 sim_size=None, position_count=None, conduit_solve_depth=1,
                 preview_interval=50,
                 test_interval=None,
                 sim_type=None,
                 load_genome_path=None,
                 game_config=None):
        """
            :param species_templates:
            :param species_configurations:
            :param sim_size:
            :param sim_type: Type of simulation (string): 2D, numerical
            """
        super(GeneticSim, self).__init__()

        self.initialized = True
        self.genetic_sim_params = genetic_sim_params
        self.steps_per_gen = time_steps_per_gen
        self.sim_size = sim_size
        self.position_count = position_count
        self.conduit_solve_depth=conduit_solve_depth
        self.sim_type = sim_type
        self.preview_interval = preview_interval
        self.test_interval = test_interval
        self.path_config = path_config
        #self.field_type = field_type  # field for 2D environments
        #self.numerical_data_type = numerical_data_type  # dataset type for numerical simulations
        #self.environment_data_file = environment_data_file
        #self.data_interval_mode = data_interval_mode
        self.environment_parameters = environment_parameters
        self.profiler = GsimProfiler()
        self.game_config = game_config

        # Initialize dictionaries for storing the most recent performance metrics
        self.test_metric = None
        self.val_metric = None

        # Create population group definitions
        self.initial_pos_sigma = []
        if not len(species_templates) == len(species_configurations):
            print("ERROR: number of species templates not equal to the number of configurations")
            self.initialized = False
            return
        self.populations = []

        if len(species_configurations) == 1:  # If there is 1 population, check that the channels are 0
            if not species_configurations[0].PD_channel == 0:
                print(
                    "Population config has PD_channel={} but there is only 1 population. Setting PD_channel to 0".format(
                        species_configurations[0].PD_channel))
                species_configurations[0].PD_channel = 0
        else:  # if there is > 1 population, make sure channels aren't duplicated
            existing_channels = []
            for i, config in enumerate(species_configurations):
                if config.PD_channel in existing_channels:
                    print("ERROR: population {} uses PD_channel {} which had already been assigned".format(i,
                                                                                                           config.PD_channel))
                    return
                existing_channels.append(config.PD_channel)

        # Initialize populations with random organisms or load based on loaded genome file
        for i in range(len(species_templates)):
            new_population = Population(species_config=species_configurations[i],
                                        species_template=species_templates[i])
            new_population.generate_random_population(initial_pos_sigma=species_configurations[i].initial_pos_sigma,
                                                      sim_size=self.sim_size, position_count=self.position_count,
                                                      randomize_genes=True,
                                                      distribution_type='Gaussian' if self.sim_type == '2D' else 'zero' if self.sim_type == 'game' else 'numerical-binary')
            self.populations.append(new_population)
            self.initial_pos_sigma.append(species_configurations[i].initial_pos_sigma)
        if load_genome_path is not None:
            self.load_genome(load_genome_path)

        # Create simulation environment
        self.env = SimEnvironment(sim_size=self.sim_size, time_steps=self.steps_per_gen,
                                  pher_no_channels=len(self.populations),
                                  PD_no_channels=len(self.populations),
                                  sim_type=self.sim_type,
                                  numerical_position_count=self.position_count)

        # If using a numerical simulation, generate the numerical grid
        # if self.sim_type == 'numerical':
        #   self.env.generate_numerical_grid(no_dims=4, no_steps=11, grid_lims=[0, 1])

        # Create visualizer object
        self.visualizer = Visualizer(no_populations=len(self.populations), resolution=(500, 500),
                                     sim_size=self.sim_size,
                                     field=self.env.field)

        # Create tensorboard summary writer
        tensorboard_label = path_config.sim_name + "sim-{}_tsteps{}".format(self.sim_type, self.steps_per_gen)
        for i, population in enumerate(self.populations):
            tensorboard_label += "_pop{}-size{}-genes{}-hidden{}".format(i + 1, population.pop_count,
                                                                         population.genes_per_organism,
                                                                         population.node_counts["hidden"])
        self.writer = SummaryWriter(logdir=self.path_config.tensorboard + "/{}-".format(
            datetime.now().strftime("%Y%m%d%H%M%S%f")) + tensorboard_label, comment=tensorboard_label)
        self.key_debug_gene_connections = None

        self.game_track = None
        self.game_stats = {"win": [0], "lose": [0], "tie": [0], "count": [0]}

    def time_step(self, t, g, mode, USE_GPU=True, organism_probe: OrganismDebugProbe = None):
        # When using a numerical simulation, calculate the position density for every population
        if self.sim_type == "numerical" or self.sim_type == "game":

            # self.env.position_density_cpu2gpu(GPU=USE_GPU)  # Reset position density array
            for population in self.populations:
                self.profiler.start_measurement("position density")
                calc_position_density(environment=self.env, population=population, channel=population.PD_channel,
                                      USE_GPU=USE_GPU)
                self.profiler.end_measurement("position density")

        # ------- SENSE ------- #
        # pos_last is updated within Sense
        for population in self.populations:
            self.profiler.start_measurement("sense")
            calc_sense(population=population, env=self.env, t_step=t, sim_type=self.sim_type, USE_GPU=USE_GPU)
            self.profiler.end_measurement("sense")

        if organism_probe:
            organism_probe.log_state(populations=self.populations, stage_label="sense", t=t, GPU=USE_GPU)

        # ------- DECIDE ------- #
        for population in self.populations:
            self.profiler.start_measurement("decision")
            if USE_GPU:
                # print("(t={}) O[0] before dec: {}".format(t, population.d_O.copy_to_host()[0,0,-10:]))
                population.calc_decision_GPU(sim_size=self.env.sim_size, sim_type=self.sim_type,
                                             conduit_solve_depth=self.conduit_solve_depth)
                # print("(t={}) O[0] after dec: {}".format(t, population.d_O.copy_to_host()[0, 0, -10:]))
            else:
                # print("(t={}) O[0] before dec: {}".format(t, population.O[0,0,-10:]))
                population.calc_decision_CPU(sim_size=self.env.sim_size, sim_type=self.sim_type,
                                             conduit_solve_depth=self.conduit_solve_depth)
                # print("(t={}) O[0] after dec: {}".format(t, population.O[0, 0, -10:]))
            self.profiler.end_measurement("decision")
            population.apply_position_mask(env=self.env, USE_GPU=USE_GPU)
            update_position_change(population, USE_GPU=USE_GPU)

        if organism_probe:
            organism_probe.log_state(populations=self.populations, stage_label="decision", t=t, GPU=USE_GPU)
            if USE_GPU:
                pos = self.populations[0].d_pos.copy_to_host()
            else:
                pos = self.populations[0].pos
            # print("({}) t = {}: {} = {}".format("GPU" if USE_GPU else "CPU", t, pos[15, :], pos[15, :].sum()))

        # ------- GAME ENVIRONMENT STATE UPDATE ------- #
        if self.sim_type == "game":
            from src.gpu.tictactoe_gpu import _count_predictions_above_threshold
            from numba import cuda

            threadsperblock = (16, 16)
            blockspergrid_x = (self.populations[0].d_O.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
            blockspergrid_y = (self.game_config.no_decisions + threadsperblock[1] - 1) // threadsperblock[1]
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            dec = np.zeros(self.game_config.no_decisions)
            d_dec = cuda.to_device(dec)

            # _count_predictions_above_threshold[blockspergrid, threadsperblock](self.populations[0].d_O, self.populations[0].decision_threshold,
            #                                                                   self.populations[0].out_offset, self.game_config.no_decisions, d_dec)

            self.populations[0].O = self.populations[0].d_O.copy_to_host()
            decisions_all = self.populations[0].O[:, 0,
                            self.populations[0].out_offset:(self.populations[0].out_offset + self.position_count)]
            decisions_int = decisions_all.argmax(1)
            decisions_int[decisions_all.max(1) <= 0] = -1

            if t == 0 and g % 100 == 0:
                self.populations[0].organisms[0].brain.draw_graph(
                    static_input_names=self.populations[0].template.input_names,
                    output_names=self.populations[0].template.output_names,
                    data_input_count=self.populations[0].template.data_input_count,
                    data_input_names=self.env.data_in_names,
                    position_density_names=self.env.position_density_names,
                    save_path="graph_temp.png"
                    )

            # Pick majority decision
            decision_mode = "consensus"
            if decision_mode == "consensus":
                unique_values, counts = np.unique(decisions_int, return_counts=True)
                max_count_index = np.argmax(counts)
                consensus_value = unique_values[max_count_index]
                # if consensus_value > 0:
                #    print("MOVE: {}".format(self.populations[0].template.output_names[consensus_value]))
                accountability_mask = (decisions_int == consensus_value).astype('int')
            else:
                # Pick random decision
                valid_decisions = np.where(decisions_int >= 0)[0]
                if len(valid_decisions) > 0:
                    dec_rand_org_idx = np.random.choice(valid_decisions)
                    # dec_rand_org_idx = 0
                    consensus_value = decisions_int[dec_rand_org_idx]
                    accountability_mask = np.zeros(len(decisions_int)).astype('int')
                    accountability_mask[dec_rand_org_idx] = 1
                else:
                    consensus_value = -1
                    accountability_mask = np.ones(len(decisions_int)).astype('int')


            is_legal_move = legal_move(game_state=self.env.data_in, move_query=consensus_value)

            self.env.data_in = self.env.d_data_in.copy_to_host()
            self.populations[0].actb_score = self.populations[0].d_actb_score.copy_to_host()
            if is_legal_move:
                self.populations[0].actb_score += accountability_mask
            else:
                self.populations[0].actb_score += accountability_mask * 10

            self.populations[0].d_actb_score = cuda.to_device(self.populations[0].actb_score)
            game_loss = not is_legal_move
            game_win = False
            game_tie = False

            if not game_loss:
                # Update player move
                self.env.data_in = update_game_state(game_state=self.env.data_in, move=consensus_value, is_player=True)
                game_result = check_winner(self.env.data_in)
                if game_result is None:
                    # Update AI move
                    self.env.data_in = update_game_state(game_state=self.env.data_in, is_player=False)
                    game_result = check_winner(self.env.data_in)
                if game_result == -1:
                    game_loss = True

                elif game_result == 0:
                    game_tie = True
                elif game_result == 1:
                    game_win = True

            self.populations[0].fitness = self.populations[0].d_fitness.copy_to_host()
            if game_loss or game_tie or game_win:
                if game_win:
                    self.game_stats["win"].append(self.game_stats["win"][-1] + 1)
                    fitness_mult = 5
                elif game_loss:
                    self.game_stats["lose"].append(self.game_stats["lose"][-1] + 1)
                    fitness_mult = -0.1
                elif game_tie:
                    self.game_stats["tie"].append(self.game_stats["tie"][-1] + 1)
                    fitness_mult = 0.1
                game_over = True
                self.game_stats["count"].append(self.game_stats["count"][-1] + 1)
                if game_win:
                    pass
                    # **** losses have the same impact on repopulation as wins, but shouldnt they weight higher if they are expected to be more rare?
            else:
                # Small reward for continuing game without illegal move
                fitness_mult = 0.1
                game_over = False

            fitness_org = self.populations[0].actb_score * fitness_mult
            self.populations[0].fitness += fitness_org
            self.populations[0].d_fitness = cuda.to_device(self.populations[0].fitness)

            self.env.data_in[self.env.data_in > 0] = 1.
            self.env.data_in[self.env.data_in < 0] = -1.
            probabilities = np.zeros(9)
            for i in range(9):
                probabilities[i] = np.array(np.array(decisions_int) == i).sum()
            self.game_track.update_state(state=np.array(self.env.data_in).copy(), probabilities=probabilities)

            if game_over:
                return "win" if game_win else "loss" if game_loss else "tie"

            self.env.d_data_in = cuda.to_device(self.env.data_in)

        # ------- FITNESS ------- #
        if not self.sim_type == 'game':
            for population in self.populations:
                self.profiler.start_measurement("fitness")
                calc_fitness(fitness_type=population.fitness_type, population=population, t=t, env=self.env,
                             USE_GPU=USE_GPU)
                # print(t, self.populations[0].d_fitness.copy_to_host().mean())
                self.profiler.end_measurement("fitness")

        if g % self.preview_interval == 0 or mode == 'test' or mode == 'validate':
            self.profiler.start_measurement("visualize")
            for p in range(len(self.populations)):
                self.visualizer.append_plot(population=self.populations[p], env=self.env, pop_idx=p, USE_GPU=USE_GPU)
            self.profiler.end_measurement("visualize")

        #print("t={}  |   change={}   |   {}".format(t, self.populations[0].d_pos_change.copy_to_host()[0], self.populations[0].d_pos.copy_to_host()[0, :]))

        return None

    def eval_step(self, data_input, USE_GPU=False):
        """
        Evaluate a single time step of the model with the specified data input
        :param data_input: data input array [N_data_in]
        :param USE_GPU: If 'False' then use the CPU version of genetic sim calculations
        :return:
        """
        self.env.data_in = np.expand_dims(data_input, 1)

        if self.sim_type == "numerical":
            for population in self.populations:
                calc_position_density(environment=self.env, population=population, channel=population.PD_channel,
                                      USE_GPU=USE_GPU)

        # ------- SENSE ------- #
        for population in self.populations:
            calc_sense(population=population, env=self.env, t_step=0, sim_type=self.sim_type, USE_GPU=USE_GPU)

        # ------- DECIDE ------- #
        for population in self.populations:
            population.calc_decision_CPU(sim_size=self.env.sim_size, sim_type=self.sim_type,
                                         conduit_solve_depth=self.conduit_solve_depth)
            population.apply_position_mask(env=self.env, USE_GPU=USE_GPU)
            update_position_change(population, USE_GPU=USE_GPU)

    def simulate_generation(self, g, gene_pool, no_generations, T, mode='train', USE_GPU=True,
                            pbar=None, ZERO_INITIAL_POS=False, visualize_config=None,
                            val_start_time_force=None, organism_probe: OrganismDebugProbe = None):
        """
        Simulate generation count 'g'. Given the current population(s), calculate the necessary number of steps for
        sensing, decision, and fitness. Finally, if not in validation mode, generate the next generation based on the
        resulting fitness.
        :param g: current generation count
        :param gene_pool: gene pool list
        :param no_generations: total number of generations
        :param T: number of time steps
        :param skip_animation: if True, skip generation of animations (speeds up training)
        :param mode: processing mode: 'train', 'test', 'validate'
        :param organism_probe: (optional) for debugging, detailed tracking using the Species.OrganismDebugProbe class
        :return:
        """

        if visualize_config is None:
            visualize_config = {"mode": "full", "animation": "False", "tensorboard": True, "performance log": False,
                                "numerical positions": True, "plotly summary": False}
        else:
            if "mode" not in list(visualize_config.keys()):
                print("WARNING: Specify \"mode\" in \"visualize_config\"")
            if "tensorboard" not in list(visualize_config.keys()):
                visualize_config["tensorboard"] = True
            if "animation" not in list(visualize_config.keys()):
                visualize_config["animation"] = False
            if "performance log" not in list(visualize_config.keys()):
                visualize_config["performance log"] = False
            if "numerical positions" not in list(visualize_config.keys()):
                visualize_config["numerical positions"] = False
            if "plotly summary" not in list(visualize_config.keys()):
                visualize_config["plotly summary"] = False

        for population in self.populations:
            population.update_mat(GPU=USE_GPU)
            population.fitness = np.ones(population.pop_count)
            population.reset_health(GPU=USE_GPU)
            population.update_organism_arrays_cpu2gpu(GPU=USE_GPU, sim_type=self.sim_type)

        # *************** Zero initial positions
        if ZERO_INITIAL_POS:
            from numba import cuda
            for pop in self.populations:
                pop.pos = np.zeros(pop.pos.shape)
                pop.pos[:, -1] = 1
                pop.d_pos = cuda.to_device(pop.pos)

        # Field generation
        if self.sim_type == '2D':
            self.env.generate_field(field_type=self.field_type,
                                    # field_seed=0 if g % preview_intervals else None,
                                    # field_seed=0,
                                    t_series_start_range=None,
                                    t_series_start_val=None,
                                    GPU_arrays=USE_GPU)
            self.visualizer.plot_field = self.env.field
            self.env.pheromone_cpu2gpu()  # Allocate pheromone arrays
            self.env.population_density_cpu2gpu()  # Allocate population density arrays
        elif self.sim_type == 'numerical':
            self.profiler.run(self.env.load_numerical_data, data_type=self.environment_parameters['data type'],
                              duration=T,
                              GPU_arrays=USE_GPU, is_validation=(mode == 'validate' or mode == 'test'),
                              populations=self.populations,
                              truncated_normal_distr_start=True,
                              start_t_force=val_start_time_force, environment_parameters=self.environment_parameters)
            self.env.position_density_cpu2gpu(GPU=USE_GPU)  # Allocate position density arrays
            if val_start_time_force is not None:  # Specified start time
                if (val_start_time_force + T) > len(self.env.env_data_file['dates']):
                    T = len(self.env.env_data_file['dates']) - val_start_time_force - 1
            if T == -1 and mode == "validate":
                # If T=-1, use the full remaining validation data samples
                T = len(self.env.env_data_file['dates']) - self.env.validation_start_sample - 1
                if pbar is None:
                    print("Running full validation span")
        elif self.sim_type == 'game':
            self.profiler.run(self.env.initialize_game_env, game_config=self.game_config, populations=self.populations,
                              GPU_arrays=USE_GPU)
            self.env.position_density_cpu2gpu(GPU=USE_GPU)  # Allocate position density arrays
            T = int(1e6)

        # Full dataset validation
        if T is None and mode == "validate":
            T = self.env.data_in.shape[1] - 15
            if pbar is None:
                print("Running full dataset validation")
        elif T is None or T == -1:
            print("ERROR: Duration is None or -1 while mode is {}".format(mode))
            return

        # ------- Dropout ------- #
        if self.genetic_sim_params.dropout > 0.:
            for p in self.populations:
                p.generate_dropout_mask(dropout_rate=self.genetic_sim_params.dropout, USE_GPU=USE_GPU)

        if organism_probe:
            organism_probe.log_parameters(populations=self.populations, GPU=USE_GPU, environment=self.env)

        # ------- Perform Simulation Time Steps ------- #
        if self.sim_type == "game":
            self.game_track = TicTacToe_animation()
        for t in tqdm(range(T)) if mode == "validate" else range(T):
            self.profiler.start_measurement("Timestep")
            status = self.time_step(t=t, g=g, mode=mode, USE_GPU=USE_GPU, organism_probe=organism_probe)
            self.profiler.end_measurement("Timestep")
            if status is not None:
                print("Stopping generation at t={} ({})".format(t, status))
                break

        # Copy GPU organism arrays to CPU
        if USE_GPU:
            for population in self.populations:
                self.profiler.run(population.update_organism_arrays_gpu2cpu)
        if mode == "validate" and pbar is None:
            print("Validation run complete. Generating figures...")
            # from src.Visualize import Create_FanChart_Plotly
            # Create_FanChart_Plotly(plot_pos=self.visualizer.plot_pos, populations=self.populations,
            #                       output_data=self.env.data_out, spread_data=self.env.spread_ratio)
        # At preview interval, generate preview plots/graphs
        if g > 0 and g % self.preview_interval == 0 or mode == "validate" or mode == "test" and not visualize_config[
                                                                                                        'mode'] == 'none':

            metrics = numerical_visualization_set(visualizer=self.visualizer, populations=self.populations,
                                                  env=self.env, config=visualize_config,
                                                  path_config=self.path_config, g=g, no_generations=no_generations,
                                                  sim_type=self.sim_type, writer=self.writer, gene_pool=gene_pool,
                                                  data_interval_mode=self.environment_parameters["data interval mode"],
                                                  mode=mode,
                                                  game_track=self.game_track)
            if mode == "validate":
                self.val_metric = metrics
            elif mode == "test":
                self.test_metric = metrics

        writer_pop_prefix = "{}-{} "  # prefix to add to tensorboard summarywriter labels
        # Log tensorboard performance statistics
        if mode == "train" and visualize_config["tensorboard"]:
            for p in range(len(self.populations)):
                self.writer.add_scalar(
                    "Train/" + writer_pop_prefix.format(p, self.populations[p].pop_label) + "Mean ROI",
                    np.mean(self.populations[p].fitness), g)
                self.writer.add_scalar(
                    "Train/" + writer_pop_prefix.format(p, self.populations[p].pop_label) + "Max ROI",
                    np.max(self.populations[p].fitness), g)
                self.writer.add_scalar(
                    "Train/" + writer_pop_prefix.format(p, self.populations[p].pop_label) + "Min ROI",
                    np.min(self.populations[p].fitness), g)
                # print("g = {}: F = {}".format(g, np.mean(self.populations[0].fitness)))
                self.writer.add_histogram(
                    tag="Train/" + writer_pop_prefix.format(p, self.populations[p].pop_label) + "ROI",
                    values=self.populations[p].fitness, global_step=g)
                if not visualize_config["mode"] == "none":
                    # Add fitness, age arrays to visualizer tracking
                    self.visualizer.average_age[p].append(np.mean(self.populations[p].age))
                    self.visualizer.fitness[p].append(self.populations[p].fitness)

            if self.key_debug_gene_connections is not None:
                self.visualizer.key_connection_genes_update(organisms=self.populations[0].organisms,
                                                            connection_list=self.key_debug_gene_connections)
        elif mode == "validate" and visualize_config["tensorboard"]:
            pass
        elif mode == 'test' and visualize_config["tensorboard"]:
            for p in range(len(self.populations)):
                self.writer.add_scalar(
                    "Test/" + writer_pop_prefix.format(p, self.populations[p].pop_label) + "Mean ROI",
                    np.mean(self.populations[0].fitness), g)
                self.writer.add_scalar("Test/" + writer_pop_prefix.format(p, self.populations[p].pop_label) + "Max ROI",
                                       np.max(self.populations[0].fitness), g)
                self.writer.add_scalar("Test/" + writer_pop_prefix.format(p, self.populations[p].pop_label) + "Min ROI",
                                       np.min(self.populations[0].fitness), g)
                self.writer.add_histogram(
                    tag="Test/" + writer_pop_prefix.format(p, self.populations[p].pop_label) + "ROI Distribution",
                    values=self.populations[0].fitness, global_step=g)

        if not mode == 'validate':
            self.visualizer.reset_population_data()

        # ------- REPOPULATE ------- #
        # Generate next generation (Repopulation/Mutation)
        if mode == 'train':
            ######################### ******************* CROSS FITNESS
            # if g == 0:
            #   print("*" * 20 + " USING CROSS FITNESS FOR POP 0 " + "*" * 20)
            for i, population in enumerate(self.populations):

                if self.genetic_sim_params.spoiled_pop_idx is not None:
                    if not i == self.genetic_sim_params.spoiled_pop_idx:
                        population.fitness = population.fitness + self.populations[
                            self.genetic_sim_params.spoiled_pop_idx].fitness.mean() * self.genetic_sim_params.spoiled_pop_gamma
                # if not i==0:
                #    population.fitness = population.fitness + self.populations[0].fitness.mean()*0.5

                if not self.genetic_sim_params.no_trade_penalty == 0. and i == 0:
                    population.fitness[population.pos_change < self.genetic_sim_params.no_trade_threshold] -= self.genetic_sim_params.no_trade_penalty

                ### ************* Apply weight decay
                if self.genetic_sim_params.weight_decay_gamma > 0.:
                    w_sum = (population.W ** 2).sum(1).sum(1)
                    population.fitness *= 1. / (1. + self.genetic_sim_params.weight_decay_gamma * w_sum)

                self.profiler.start_measurement("repopulation")
                elite_count = self.genetic_sim_params.elite_count

                gene_pool, sel_count = tournament_repopulate(mutation_rate=self.genetic_sim_params.mutation_rate,
                                                             population=population,
                                                             initial_pos_sigma=self.initial_pos_sigma[i],
                                                             tournament_size=self.genetic_sim_params.tournament_size,
                                                             crossover_rate=self.genetic_sim_params.crossover_rate,
                                                             elite_count=elite_count,
                                                             sim_size=self.sim_size,
                                                             skip_rate=self.genetic_sim_params.gene_skip_rate,
                                                             dupl_rate=self.genetic_sim_params.gene_dupl_rate,
                                                             sim_type=self.sim_type,
                                                             position_count=self.position_count,
                                                             tie_breaker=None)
                if not visualize_config['mode'] == 'none':
                    self.visualizer.genetic_diversity[i].append(genetic_diversity(gene_pool))
                    self.visualizer.org_selection_count[i].append(sel_count)
                self.profiler.end_measurement("repopulation")
        if mode == 'train':
            if pbar is None:
                print("Generation {} of {}    |    Average Fitness: {:0.4f}".format(g + 1, no_generations,
                                                                                    np.mean(
                                                                                        self.populations[0].fitness)))
            else:
                try:  ################################### SHOULD FIX THIS
                    pbar.set_postfix(avg_fitness=np.mean(self.populations[0].fitness),
                                     test_fitness=self.test_metric[0]["ROI/BAH_ROI Mean"] if
                                     self.test_metric is not None and not len(self.test_metric) == 0 else 0)
                except Exception as e:
                    pass
        elif mode == 'test':
            if pbar is None and self.test_metric is not None:
                print("Test       |       BAH ROI Comparison Mean: {:0.4f}".format(
                    self.test_metric[0]["ROI/BAH_ROI Mean"]))

        return gene_pool

    def train(self, no_generations, USE_GPU=True):
        print("Starting training for {} generations".format(no_generations))
        gene_pool = None

        # Generate parameters table for simulation
        parameters_str = "Training data file: " + self.environment_parameters['data file'] + "\n"
        parameters_str += self.summarize_generation_sim_parameters(no_generations, self.steps_per_gen, "Train",
                                                                  True,False, print_str=True)
        parameters_str += self.genetic_sim_params.parameter_summary(print_str=True)
        for population in self.populations:
            parameters_str += population.parameter_summary(print_str=True)
        with open(self.path_config.gs_state_dir + "/training_parameters.txt", 'w') as file:
            file.write(parameters_str)

        pbar = tqdm(range(no_generations), desc="Train")
        for g in pbar:
            self.profiler.start_measurement("train generation")
            gene_pool = self.simulate_generation(g, gene_pool, no_generations, T=self.steps_per_gen,
                                                 visualize_config={"mode": "full", "numerical positions": True},
                                                 pbar=pbar, USE_GPU=USE_GPU)
            self.profiler.end_measurement("train generation")
            if self.test_interval is not None:
                if g % self.test_interval == 0 and g > 1:
                    self.profiler.run(self.test, time_steps=self.steps_per_gen, g=g, pbar=pbar)
        self.profiler.report(save_report_path=self.path_config.gs_state_dir + "/training_profiling.txt")

    def test(self, time_steps, g, pbar=None):
        """
        Run test during training on the current populations. This will generate the associated sample plots/graphs.
        :param time_steps:
        :return:
        """
        gene_pool = None
        if time_steps is None:
            time_steps = self.steps_per_gen
        self.simulate_generation(g, gene_pool, no_generations=1, T=time_steps, visualize_config={"mode": "full"},
                                 mode='test',
                                 pbar=pbar)
        if g > 0:
            self.save_genome(self.path_config.gs_state_dir + "GS_state.p",
                             self.path_config.gs_state_dir + "config.json", g)
            with open(self.path_config.gs_state_dir + "test_metrics.json", "w") as write_file:
                json.dump(self.test_metric, write_file, indent=4)

    def final_validation(self, time_steps=None, USE_GPU=True, FULL_VALIDATION=False, ZERO_INITIAL_POS=False,
                         SAVE_METRICS=False, SAVE_VALIDATION_RUN=False, validation_dataset_name=None,
                         visualization_config=None, start_time_force=None, organism_probe=None):
        """
        Run final validation on the current populations. This will generate the associated sample plots/graphs.
        :param ZERO_INITIAL_POS:
        :param FULL_VALIDATION:
        :param USE_GPU:
        :param SAVE_METRICS:
        :param validation_dataset_name:
        :param time_steps:
        :param organism_probe: (optional) for debugging, detailed tracking using the Species.OrganismDebugProbe class
        :return:
        """
        if not FULL_VALIDATION:
            print("Starting validation for {} time steps".format(time_steps))
        if not USE_GPU:
            print("Evaluating on CPU")
        gene_pool = None
        # If time steps not specified and not using full validation, validate across the remaining samples
        if time_steps is None and not FULL_VALIDATION:
            time_steps = -1  # time_steps will be set after loading the dataset file

        if visualization_config is None:
            visualization_config = {"mode": "full", "numerical positions": True}

        self.simulate_generation(0, gene_pool, 1, T=time_steps, mode='validate', USE_GPU=USE_GPU,
                                 visualize_config=visualization_config, ZERO_INITIAL_POS=ZERO_INITIAL_POS,
                                 val_start_time_force=start_time_force, organism_probe=organism_probe)
        if SAVE_VALIDATION_RUN:
            import pickle
            from src.fitness import calc_fitness_roi_validation
            # Calculate final ROI for each organism
            val_positions = np.array(self.visualizer.plot_pos)  # dimensions: [population, t, org, pos]
            val_positions = val_positions / np.tile(np.expand_dims(np.sum(val_positions, 3), 3),
                                                    (1, 1, 1, val_positions.shape[3]))

            org_roi = []
            for p in range(len(self.populations)):
                org_roi.append([])
                for i_org in range(self.populations[p].pop_count):
                    pos = val_positions[p, :, i_org, :]
                    fitness = calc_fitness_roi_validation(data_charts=self.env.data_out, positions=pos,
                                                          fee_pct=self.populations[p].trans_fee_pct,
                                                          fee_discount_idx=self.populations[p].trans_fee_discount_idx,
                                                          spread_ratio=self.env.spread_ratio, debug=False)
                    org_roi[p].append(fitness[-1])
            # Calculate ROI for average decision
            val_positions_avg = np.mean(val_positions, 2)
            val_positions_avg = val_positions_avg / np.tile(np.expand_dims(np.sum(val_positions_avg, 2), 2),
                                                            (1, 1, val_positions.shape[3]))
            pop_avg_roi = []
            for p in range(len(self.populations)):
                fitness_avg = calc_fitness_roi_validation(data_charts=self.env.data_out, positions=val_positions_avg[p],
                                                          fee_pct=self.populations[p].trans_fee_pct,
                                                          fee_discount_idx=self.populations[p].trans_fee_discount_idx,
                                                          spread_ratio=self.env.spread_ratio, debug=False)
                pop_avg_roi.append(fitness_avg)

            # Save population positions for validation run to pickle file
            validation_run_data = {'population final ROI': np.array(org_roi),
                                   'population average ROI': np.array(pop_avg_roi),
                                   'population average final ROI': np.array(pop_avg_roi)[:, -1],
                                   'population average positions': np.array(val_positions_avg),
                                   'timestamp': datetime.utcnow(),
                                   'dataset': validation_dataset_name}
            pickle.dump(validation_run_data, open(self.path_config.gs_state_dir + "validation_run_data.p", 'wb'))
            try:
                self.val_metric.append({"Validation dataset": validation_dataset_name})
            except:
                self.val_metric["Validation dataset"] = validation_dataset_name
        if SAVE_METRICS:
            self.val_metric['Validation Dataset'] = validation_dataset_name
            if self.visualizer.plot_pos:
                threshold_percent = 5  # minimum percent change for detection
                self.val_metric['Position Change Count'] = dict()
                for i, pos in enumerate(self.visualizer.plot_pos):
                    pos_avg = np.array(pos).mean(1)  # average position for all organisms
                    change_count = 0
                    for t in range(1, pos_avg.shape[0]):
                        percent_change = np.abs(pos_avg[t] - pos_avg[t - 1]) / (np.abs(pos_avg[t - 1] + 1)) * 100
                        change_count += np.sum(percent_change > threshold_percent)
                    self.val_metric['Position Change Count']['{}'.format(i)] = "{}".format(change_count)
                print("Position Change Count:", self.val_metric['Position Change Count'])
            with open(self.path_config.gs_state_dir + "validation_metrics.json", "w") as write_file:
                json.dump(self.val_metric, write_file, indent=4)

    def save_genome(self, path, config_path=None, generations=0):
        import pickle
        import inspect
        from datetime import datetime
        # for each organism in each population, store all genes
        # shape: [no. population][no. organisms][no. genes]
        genomes = []
        for i_pop in range(len(self.populations)):
            population_genome = []
            for i_org in range(self.populations[i_pop].pop_count):
                organism_genome = []
                for g in self.populations[i_pop].organisms[i_org].genes:
                    organism_genome.append(g)
                population_genome.append(organism_genome)
            genomes.append(population_genome)

        config = dict()
        config['mutation rate'] = self.genetic_sim_params.mutation_rate
        config['crossover rate'] = self.genetic_sim_params.crossover_rate
        config['tournament size'] = self.genetic_sim_params.tournament_size
        config['elite count'] = self.genetic_sim_params.elite_count
        config['gene duplication rate'] = self.genetic_sim_params.gene_dupl_rate
        config['gene skip rate'] = self.genetic_sim_params.gene_skip_rate
        config['position count'] = self.position_count
        config['training time steps'] = self.steps_per_gen
        config["dataset"] = os.path.basename(self.environment_parameters["data file"])
        config['populations'] = []

        for i_pop in range(len(self.populations)):
            pop_config = dict()
            pop_config['population count'] = self.populations[i_pop].pop_count
            pop_config['decision threshold'] = self.populations[i_pop].decision_threshold
            pop_config['genes per organism'] = self.populations[i_pop].genes_per_organism
            pop_config['internal neuron count'] = self.populations[i_pop].node_counts["hidden"]
            pop_config['conduit neuron count'] = self.populations[i_pop].node_counts["conduit"]
            pop_config['memory neuron count'] = self.populations[i_pop].node_counts["memory"]
            pop_config['population node count'] = self.populations[i_pop].population_node_count
            pop_config['max CLK period'] = self.populations[i_pop].max_CLK_period
            pop_config['max RR period'] = self.populations[i_pop].max_RR_period
            pop_config['max connections'] = self.populations[i_pop].max_connections
            pop_config['pher channel'] = self.populations[i_pop].pher_channel
            pop_config['PD channel'] = self.populations[i_pop].PosD_channel
            pop_config['position names'] = self.populations[i_pop].template.position_names
            pop_config['data input count'] = self.populations[i_pop].template.data_input_count
            pop_config['fitness type'] = self.populations[i_pop].fitness_type
            pop_config['use transaction fee'] = self.populations[i_pop].use_trans_fee
            pop_config['template'] = dict()
            for name in dir(self.populations[i_pop].template):
                value = getattr(self.populations[i_pop].template, name)
                if not name.startswith('__') and not inspect.ismethod(value):
                    pop_config['template'][name] = value
            pop_config['pop label'] = self.populations[i_pop].pop_label
            pop_config['fan color'] = list(self.populations[i_pop].fan_color.astype('float'))
            pop_config['line color'] = list(self.populations[i_pop].line_color.astype('float'))
            pop_config['use spread'] = self.populations[i_pop].use_spread
            pop_config['fitness cap'] = self.populations[i_pop].fitness_cap
            config['populations'].append(pop_config)

        config['environment'] = {'data input names': self.env.data_in_names,
                                 'position density names': self.env.position_density_names}

        timestamp = datetime.now().isoformat()
        # print("Saving genome with timestamp {}".format(timestamp))
        pickle.dump([genomes, timestamp, generations], open(path, 'wb'))
        if config_path is not None:
            with open(config_path, "w") as outfile:
                json.dump(config, outfile)

    def load_genome(self, path):
        import pickle
        # genome: [no. population][no. organisms][no. genes]
        [genome, timestamp, generations] = pickle.load(open(path + "/GS_state.p", "rb"))
        print("Loading model {} with {} generations".format(timestamp, generations))
        for i_pop in range(len(self.populations)):
            for i_org in range(self.populations[i_pop].pop_count):
                self.populations[i_pop].organisms[i_org].load_genes(genes=genome[i_pop][i_org], clear_connections=True)

    def save_state(self, path, GPU=True):
        import pickle
        states = []
        positions = []
        clk_lims = []
        clk_counters = []
        clk_sigs = []
        rr_lims = []
        rr_counters = []
        for population in self.populations:
            if GPU:
                population.d_O.copy_to_host(population.O)
                population.d_pos.copy_to_host(population.pos)

                population.d_clk_lim.copy_to_host(population.clk_lim)
                population.d_clk_counter.copy_to_host(population.clk_counter)
                population.d_clk_sig.copy_to_host(population.clk_sig)

                population.d_rr_lim.copy_to_host(population.rr_lim)
                population.d_rr_counter.copy_to_host(population.rr_counter)

            states.append(population.O)
            positions.append(population.pos)
            clk_lims.append(population.clk_lim)
            clk_counters.append(population.clk_counter)
            clk_sigs.append(population.clk_sig)
            rr_lims.append(population.rr_lim)
            rr_counters.append(population.rr_counter)

        #pickle.dump([states, positions, [clk_lims, clk_counters, clk_sigs], [rr_lims, rr_counters]], open(path, "wb"))
        pickle.dump({"state": states, "pos": positions,
                     "clk": {"lim": clk_lims, "counter": clk_counters, "sig": clk_sigs},
                     "rr": {"lim": rr_lims, "counter": rr_counters}},
                     open(path, "wb"))

    def load_state(self, path, GPU=True):
        import pickle
        #[states, positions, CLK, RR] = pickle.load(open(path, "rb"))
        prev_state = pickle.load(open(path, "rb"))
        for n, population in enumerate(self.populations):
            population.O = prev_state["state"][n]
            population.pos = prev_state["pos"][n]
            population.clk_lim = prev_state["clk"]["lim"][n]
            population.clk_counter = prev_state["clk"]["counter"][n]
            population.clk_sig = prev_state["clk"]["sig"][n]
            population.rr_lim = prev_state["rr"]["lim"][n]
            population.rr_counter = prev_state["rr"]["counter"][n]
            if GPU:
                from numba import cuda
                population.d_O = cuda.to_device(population.O)
                population.d_pos = cuda.to_device(population.pos)
                population.d_clk_lim = cuda.to_device(population.clk_lim)
                population.d_clk_counter = cuda.to_device(population.clk_counter)
                population.d_clk_sig = cuda.to_device(population.clk_sig)
                population.d_rr_lim = cuda.to_device(population.rr_lim)
                population.d_rr_counter = cuda.to_device(population.d_rr_counter)

    def summarize_generation_sim_parameters(self, no_generations, T, mode, gpu, zero_init_pos,
                                            print_str=False):
        out_str = "{:-^70}\n".format("Generation Simulation Parameters")
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Parameter", "Value")
        out_str += 70 * "-" + "\n"
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("# Generations", no_generations)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Generation Time Steps", T)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Simulation Mode", mode)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("GPU", gpu)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Conduit Solve Depth", self.conduit_solve_depth)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Zero Initial Position", zero_init_pos)
        out_str += "-" * 70 + "\n"
        if print_str:
            print(out_str)
        return out_str


class GeneticSimParameters:
    def __init__(self, mutation_rate=1e-2, crossover_rate=0.5, tournament_size=2, elite_count=3, gene_skip_rate=0.,
                 gene_dupl_rate=0., no_trade_threshold=0, no_trade_penalty=0.,
                 weight_decay_gamma=0., dropout=0., spoiled_pop_idx=0, spoiled_pop_gamma=0.):
        super(GeneticSimParameters, self).__init__()
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.gene_skip_rate = gene_skip_rate
        self.gene_dupl_rate = gene_dupl_rate
        self.no_trade_threshold = no_trade_threshold
        self.no_trade_penalty = no_trade_penalty
        self.weight_decay_gamma = weight_decay_gamma
        self.dropout = dropout
        self.spoiled_pop_idx = spoiled_pop_idx
        self.spoiled_pop_gamma = spoiled_pop_gamma

    def parameter_summary(self, print_str=False):
        out_str = "{:-^70}\n".format("Genetic Algorithm Parameters")
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Parameter", "Value")
        out_str += 70 * "-" + "\n"
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Mutation Rate", self.mutation_rate)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Crossover Rate", self.crossover_rate)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Tournament Size", self.tournament_size)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Elite Count", self.elite_count)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Gene Skip Rate", self.gene_skip_rate)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Gene Duplication Rate", self.gene_dupl_rate)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Weight Decay Gamma", self.weight_decay_gamma)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Dropout Factor", self.dropout)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Spoiled Population Index", self.spoiled_pop_idx if self.spoiled_pop_idx is not None else "None")
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Spoiled Population Gamma", self.spoiled_pop_gamma if self.spoiled_pop_gamma is not None else "None")
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("No Trade Penalty", self.no_trade_penalty)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("No Trade Threshold", self.no_trade_threshold)
        out_str += "-" * 70 + "\n"
        if print_str:
            print(out_str)
        return out_str


def load_trader_model(load_model_folder, environment_params=None, genetic_params=None, sim_name=None,
                      init_CPU_arrays=False, position_mask=None, initial_state=None):
    """
    Load the configuration and parmaters from a saved model. If "sim_name" not specified, use the load model name as
    the new model save name path.
    :param model_folder:
    :param environment_params:
    :param genetic_params:
    :param sim_name:
    :param init_CPU_arrays: set to 'True' if running in CPU inference mode (not GPU)
    :param position_mask: masks for disabling certain output positions
    :param initial_state: population state object for setting the initial states to match a previous run
    :return:
    """
    from species_config.species_trader import initialize_trader
    from src.species import PopulationConfig
    from utils.path_config import PathConfig

    config = json.load(open(load_model_folder + "/config.json", "rb"))

    species_configs = []
    templates = []
    for ip in range(len(config["populations"])):
        pop_params = config["populations"][ip]
        species_config = PopulationConfig(config_dict=pop_params)
        species_configs.append(species_config)
        if environment_params is None:
            trader = initialize_trader(pickle_data_file=None, no_populations=len(config["populations"]),
                                       memory_node_count=pop_params["memory neuron count"],
                                       currencies=pop_params['position names'],
                                       no_data_inputs=pop_params['data input count'])
        else:
            trader = initialize_trader(pickle_data_file=environment_params['data file'],
                                       no_populations=len(species_configs),
                                       memory_node_count=species_configs[0].memory_neuron_count)
        templates.append(trader)

    if sim_name is None:
        sim_name = load_model_folder.split('/')[-2] if load_model_folder[-1] == '/' else load_model_folder.split('/')[-1]
    path_config = PathConfig(sim_name=sim_name, populations=species_configs, root_dir="./")

    if genetic_params is None:
        genetic_params = GeneticSimParameters(mutation_rate=config["mutation rate"],
                                              crossover_rate=config["crossover rate"],
                                              tournament_size=config["tournament size"],
                                              elite_count=config["elite count"],
                                              gene_dupl_rate=config["gene duplication rate"],
                                              gene_skip_rate=config["gene skip rate"])

    Gsim = GeneticSim(species_templates=templates, species_configurations=species_configs,
                      genetic_sim_params=genetic_params, time_steps_per_gen=500,
                      position_count=config['position count'],
                      path_config=path_config,
                      preview_interval=500, test_interval=500, sim_type="numerical",
                      load_genome_path=load_model_folder,
                      environment_parameters=environment_params)
    #Gsim.load_genome(path=path_config.gs_state_dir)

    if position_mask is not None:
        Gsim.env.position_mask = position_mask

    if init_CPU_arrays:
        # Initialize CPU arrays
        for ip in range(len(config["populations"])):
            Gsim.populations[ip].update_organism_arrays_cpu2gpu(GPU=False)
            Gsim.populations[ip].update_mat(GPU=False)
        Gsim.env.position_density_cpu2gpu(GPU=False)

    # load initial positions and states
    if initial_state is not None:
        pos_initial = initial_state['pos']
        state_initial = initial_state['state']
        for i_pop in range(len(Gsim.populations)):
            Gsim.populations[i_pop].pos = pos_initial[i_pop]
            Gsim.populations[i_pop].O = state_initial[i_pop]
            if not init_CPU_arrays:
                Gsim.populations[i_pop].update_organism_arrays_cpu2gpu(GPU=True)

    return Gsim


if __name__ == "__main__":
    import os
    from src.species import PopulationConfig
    from species_config.species_trader import initialize_trader
    from utils.path_config import PathConfig

    os.makedirs("outputs/", exist_ok=True)
    sample_dataset = "datasets/crypto_dataset_3minute_2024.p"

    environment_params = {"data type": "numerical-data-file",
                          "data file": sample_dataset,
                          "data interval mode": "intraday"}

    spec_config = PopulationConfig(pop_count=100, genes_per_organism=32,
                                 internal_neuron_count=20, conduit_neuron_count=8, memory_neuron_count=2,
                                 population_node_count=4,
                                 decision_threshold=0.95, max_health=10000, min_health=-10000, start_health=50.,
                                 fitness_type='ROI-numerical', max_CLK_period=500, max_RR_period=800,
                                 max_connections=500,
                                 pher_channel=0, PD_channel=0,
                                 use_spread=False, use_trans_fee=False,
                                 pop_label="W", fan_color=(0, 0, 1), line_color=(0, 0.5, .7), fitness_cap=1.04)

    trader = initialize_trader(pickle_data_file=sample_dataset, no_populations=1,
                               memory_node_count=spec_config.memory_neuron_count)

    path_config = PathConfig(sim_name="debug", populations=[spec_config], root_dir="outputs/")
    print("Saving samples to \"{}\"".format(path_config.output_dir))

    Gsim = GeneticSim(species_templates=[trader], species_configurations=[spec_config],
                      genetic_sim_params=GeneticSimParameters(), time_steps_per_gen=500,
                      position_count=trader.position_count,
                      path_config=path_config,
                      preview_interval=5, test_interval=5, sim_type="numerical",
                      load_genome_path=None,
                      environment_parameters=environment_params, conduit_solve_depth=1)

    try:
        from numba.cuda import detect
        detect()
        use_gpu = True
    except Exception as e:
        print("CUDA not detected. Using CPU.")
        use_gpu = False

    Gsim.train(no_generations=10, USE_GPU=use_gpu)