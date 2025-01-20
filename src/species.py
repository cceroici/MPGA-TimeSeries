import numpy as np
from numba import cuda
import math
from src.cortex import Brain
from src.gpu.array_gpu import _matmul, flatten_cuda_array, _copy_2d_array, _zero_and_copy_2d_array, _gated_copy_2d_array, _average_broadcast, _self_matrix_elementwise_multiply
from src.cpu.array_cpu import matmul, flatten
from src.decide import calc_decide
from src.genome import random_gene, decode_clk_init_gene, decode_rr_init_gene, decode_decision_threshold_gene, gene_codes, gene_type_labels
from src.environment import initial_positions, _find_closest_2d, _find_closest_2d_species
from src.solver import get_organism_pos
from src.position_mask import apply_mask
from tqdm import tqdm


class Species:
    """
    Species template class
    """

    def __init__(self):
        super(Species, self).__init__()

        self.name = None
        self.input_names = None
        self.output_names = None
        self.input_count = 0
        self.output_count = 0
        self.decision_indexes = dict()
        self.input_indexes = dict()
        self.data_input_count = 0
        self.static_input_count = 0
        self.data_input_start_idx = None
        self.position_density_start_idx = None

    def Initialize(self, last_fixed_input_idx, position_count=None, no_pos_density_ch=0):
        self.static_input_count = len(self.input_names)
        self.input_count = self.static_input_count + self.data_input_count
        self.data_input_start_idx = last_fixed_input_idx + 1

        if position_count is not None:  # position density (1 and 0 states) for each channel
            self.input_count += position_count * 2 * no_pos_density_ch
            self.position_density_start_idx = self.data_input_start_idx + self.data_input_count

        self.output_count = len(self.output_names)

        for i, name in enumerate(self.input_names):
            self.input_indexes[name] = i
        for i, name in enumerate(self.output_names):
            self.decision_indexes[name] = i


class Organism_Numerical:
    def __init__(self, start_position, gene_count, internal_neuron_count, conduit_neuron_count, memory_neuron_count,
                 max_connections, input_count=None, output_count=None, decision_thresh=0.95):
        super(Organism_Numerical, self).__init__()

        self.pos = start_position
        self.age = 0
        self.alive = True
        self.genes = []
        self.clk_counter = 0
        self.clk_limit = 1
        self.clk_sig = 0
        self.decision_threshold = 0
        self.decision_threshold_base = decision_thresh
        # Reaction rate
        self.rr_counter = 0
        self.rr_limit = 0
        self.gene_count = gene_count
        self.movement_speed = 1
        node_counts = {"input": input_count, "hidden": internal_neuron_count, "conduit": conduit_neuron_count,
                       "output": output_count, "memory": memory_neuron_count}
        node_counts['total'] = np.array([node_counts[key] for key in list(node_counts)]).sum()
        self.brain = Brain(node_counts=node_counts, max_connections=max_connections)

    def randomize_genes(self):

        for i in range(self.gene_count):
            self.genes.append(random_gene(node_counts=self.brain.node_counts, type="numerical"))
        self.load_genes()

    def get_genome(self):
        genome = ""
        for g in self.genes:
            genome += g
        return genome

    def load_genes(self, genes=None, clear_connections=False):
        if genes is not None:
            self.genes = genes
        else:
            genes = self.genes
        genes_no_dupl = []
        for g in genes:
            if g not in genes_no_dupl:
                genes_no_dupl.append(g)

        # Sort genes list based on gene type (smallest to largest) so that influence connection geneses are applied last
        sorted(genes, key=lambda x: int(x[0], 16))

        if clear_connections:
            self.brain.connections = []

        timer_period_count = 0
        rr_period_count = 0
        dec_thresh_gene_count = 0
        self.clk_limit = 0
        self.rr_limit = 0
        self.decision_threshold = 0
        for g in genes_no_dupl:
            if g[0] in ['0', '4', '5', '6', '7']:  # connection gene
                self.brain.add_connection(g)
            elif g[0] == '1':  # internal timer initial period gene
                self.set_internal_timer_period(g)
                timer_period_count += 1
            elif g[0] == '3':  # reaction rate initial period gene
                self.set_reaction_rate_period(g)
                rr_period_count += 1
            elif g[0] == '8':  # decision threshold init gene
                self.set_decision_threshold(g)
                dec_thresh_gene_count += 1
            elif g[0] == 'f':
                pass
            else:
                print("ERROR (load_genes()): Unknown gene code \"{}\"".format(g[0]))

        # If more than one hyperparameter gene found, average them
        if timer_period_count > 0:
            self.clk_limit = self.clk_limit / timer_period_count
        if rr_period_count > 0:
            self.rr_limit = self.rr_limit / rr_period_count
        if dec_thresh_gene_count > 0:
            self.decision_threshold = self.decision_threshold / dec_thresh_gene_count
        else:
            self.decision_threshold = self.decision_threshold_base
        # self.brain.Prune_Connections()

    def set_internal_timer_period(self, gene):
        """
        Set internal clock timer period (clk_limit). Gene value ranges from 0->255, this is scaled to 1->35
        :param gene:
        :return:
        """
        if not gene[0] == '1':
            print("Error: Attempted to import timer period gene with incorrect type code: {}".format(gene[0]))
            return
        # Add to clk limit to accumulate all CLK init genes in the genome (rather than override earlier genes)
        self.clk_limit += decode_clk_init_gene(gene_code=gene)[0]

    def set_decision_threshold(self, gene):
        """
        Set internal clock timer period (clk_limit). Gene value ranges from 0->255, this is scaled to 1->35
        :param gene:
        :return:
        """
        if not gene[0] == '8':
            print("Error: Attempted to decision threshold gene with incorrect type code: {}".format(gene[0]))
            return
        # Add to clk limit to accumulate all CLK init genes in the genome (rather than override earlier genes)
        self.decision_threshold += decode_decision_threshold_gene(gene_code=gene)[0]

    def set_reaction_rate_period(self, gene):
        if not gene[0] == '3':
            print("Error: Attempted to import reaction rate period gene with incorrect type code: {}".format(gene[0]))
            return
        # Add to rr period limit to accumulate all rr period init genes in the genome (rather than override earlier genes)
        self.rr_limit += decode_rr_init_gene(gene_code=gene)[0]


    def get_gene_types(self):
        gene_counts = np.zeros(len(gene_codes))
        for g in self.genes:
            for i in gene_codes:
                if g[0] == gene_counts[i]:
                    gene_counts[i] += 1
                    break
        return gene_counts, gene_type_labels

    def log_activity(self, log_path, input_names, output_names, organism_state, inputs=None):
        import os

        fopen_mode = 'w'
        if os.path.exists(log_path):
            fopen_mode = 'a'

        with open(log_path, fopen_mode) as f:
            f.write(25 * '*' + '\n')
            f.write(10 * ' ' + 'Age {}'.format(self.age) + 10 * ' ' + "\n")
            f.write(25 * '*' + '\n')
            f.write("\n")
            f.write(10 * "-" + "Connection " + 10 * "-" + "\n")
            f.write(self.brain.summarize_connections(print_log=False))
            f.write("\n")
            f.write(10 * "-" + "Inputs " + 10 * "-" + "\n")
            for i in range(self.brain.node_counts["input"]):
                f.write("({}) {}".format(i, input_names[i]))
                if inputs is not None:
                    f.write(" = {}\n".format(inputs[i]))
                else:
                    f.write("\n")
            f.write("\n")
            f.write(10 * "-" + "Conduit Nodes " + 10 * "-" + "\n")
            for i in range(self.brain.node_counts["input"], self.brain.node_counts["input"] + self.brain.node_counts["conduit"]):
                f.write("({}) Node {} = {}\n".format(i, i - self.brain.node_counts["input"], organism_state[i]))
            f.write("\n")
            f.write(10 * "-" + "Internal Nodes " + 10 * "-" + "\n")
            for i in range(self.brain.node_counts["input"] + self.brain.node_counts["conduit"],
                           self.brain.node_counts["input"] + self.brain.node_counts["conduit"] + self.brain.node_counts["internal"] + self.brain.node_counts["memory"]):
                f.write("({}) Node {} = {}\n".format(i, i - self.brain.node_counts["input"] - self.brain.node_counts["conduit"],
                                                     organism_state[i]))
            f.write("\n")
            f.write(10 * "-" + "Outputs " + 10 * "-" + "\n")
            for i in range(self.brain.node_counts["input"] + self.brain.node_counts["conduit"] + self.brain.node_counts["internal"] + self.brain.node_counts["memory"],
                           self.brain.node_counts["total"]):
                f.write("({}) {} = {}\n".format(i,
                                                output_names[
                                                    i - (self.brain.node_counts["conduit"] + self.brain.node_counts["internal"] + self.brain.node_counts["memory"])],
                                                organism_state[i]))
            f.write("\n")


class Population:
    def __init__(self, pop_count=None, internal_neuron_count=None, conduit_neuron_count=None, memory_neuron_count=None,
                 max_connections=None, population_node_count=None,
                 genes_per_organism=None,
                 start_health=1, max_health=None, min_health=None, vision_radius=None, species_template=None,
                 species_config=None, pher_channel=None,
                 eat_radius=None, fitness_cap=None):
        """
        Node order:
        [input, conduit, hidden, memory, output]

        :param pop_count:
        :param internal_neuron_count:
        :param conduit_neuron_count:
        :param memory_neuron_count:
        :param max_connections:
        :param population_node_count:
        :param genes_per_organism:
        :param start_health:
        :param max_health:
        :param min_health:
        :param vision_radius:
        :param species_template:
        :param species_config:
        :param pher_channel:
        :param eat_radius:
        :param fitness_cap:
        """
        super(Population, self).__init__()

        self.organisms = []
        self.template = species_template
        # pheromone parameters
        self.pher_max_radius = 30
        self.pher_min_radius = 2
        self.pher_saturation = None

        if species_template is not None:
            input_count = species_template.input_count
            output_count = species_template.output_count
        else:
            input_count = species_template.input_count
            output_count = species_template.output_count

        if species_config is not None:
            self.pop_count = species_config.pop_count
            internal_neuron_count = species_config.internal_neuron_count
            conduit_neuron_count = species_config.conduit_neuron_count
            memory_neuron_count = species_config.memory_neuron_count
            self.population_node_count = species_config.population_node_count
            self.max_connections = species_config.max_connections
            self.genes_per_organism = species_config.genes_per_organism
            self.start_health = species_config.start_health
            self.vision_radius = species_config.vision_radius  # predator-prey vision radius
            self.max_mov_speed = species_config.max_mov_speed
            self.max_CLK_period = species_config.max_CLK_period
            self.max_RR_period = species_config.max_RR_period
            self.pher_channel = species_config.pher_channel
            self.PD_channel = species_config.PD_channel  # population density channel
            self.PosD_channel = species_config.PD_channel  # position density channel
            self.max_health = species_config.max_health
            self.min_health = species_config.min_health
            if self.template.name == "predator":
                self.eat_radius = species_config.eat_radius
            self.use_spread = species_config.use_spread
            self.use_trans_fee = species_config.use_trans_fee
            self.decision_threshold = species_config.decision_threshold
            self.fitness_type = species_config.fitness_type
            self.pop_label = species_config.pop_label
            self.fan_color = species_config.fan_color
            self.line_color = species_config.line_color
            self.fitness_cap = species_config.fitness_cap
        else:
            self.pop_count = pop_count
            self.max_connections = max_connections
            self.genes_per_organism = genes_per_organism
            self.population_node_count = population_node_count
            self.start_health = start_health
            self.vision_radius = vision_radius  # predator-prey vision radius
            self.pher_channel = pher_channel
            self.max_health = max_health
            self.min_health = min_health
            if self.template.name == "predator":
                self.eat_radius = eat_radius
            self.use_spread = False
            self.use_trans_fee = False
            self.fitness_cap = fitness_cap

        self.node_counts = {"input": input_count, "conduit": conduit_neuron_count,
                            "hidden": internal_neuron_count, "memory": memory_neuron_count,
                            "output": output_count}
        self.node_names = []
        for i in range(self.template.input_count):
            self.node_names.append("input {}".format(i))
        for i in range(conduit_neuron_count):
            self.node_names.append("conduit {}".format(i))
        for i in range(internal_neuron_count):
            self.node_names.append("hidden {}".format(i))
        for i in range(memory_neuron_count):
            self.node_names.append("memory {}".format(i))
        for node_name in self.template.output_names:
            self.node_names.append("output " + node_name)

        self.node_counts["total"] = np.array([self.node_counts[k] for k in self.node_counts.keys()]).sum()
        self.out_offset = self.node_counts["total"] - self.node_counts["output"]

        # If not None, save the no inputs/outputs for plotting the debug diagram for the specified target org index
        self.debug_target_org = None
        self.debug_pos = None
        self.debug_input_states = None
        self.debug_output_states = None
        self.debug_field = None

        # Organism arrays
        self.pos = []  # organism positions
        self.pos_last = []  # organism last positions
        self.age = []  # organism ages
        self.clk_counter = []
        self.clk_lim = []
        self.clk_sig = []
        self.rr_counter = []
        self.rr_lim = []
        self.health = []
        self.alive = []
        self.fitness = np.ones(self.pop_count)
        self.mov_speed = []
        self.pher_rel = []  # pheromone release tracking
        self.NN_idx = []  # index of NN organism
        self.NN_dist = []  # distance to NN organism
        self.actb_score = []  # fitness accountability score
        self.thresh = []  # organism-specific decision threshold levels
        self.pos_change = []

        if self.template.name == "predator":
            self.prey_detected = None
            self.prey_dir = None

        # Calculation matrices
        self.O = []  # organism state matrix  [Pop count, 1, N states]
        self.W = []  # orgnaism weight matrix   [Pop count, N states, N states]
        self.B = []  # organism bias matrix  [Pop count, 1, N states]
        self.dropout_mask = []  # population node dropout mask (1 or 0) [Pop count, 1, # nodes] (0 means inactive node)

        # GPU arrays
        self.d_pos = None  # organism positions
        self.d_pos_last = None  # organism last positions
        self.d_age = None  # organism ages
        self.d_clk_counter = None
        self.d_clk_lim = None
        self.d_clk_sig = None
        self.d_rr_counter = None
        self.d_rr_lim = None
        self.d_health = None
        self.d_alive = None
        self.d_fitness = None
        self.d_mov_speed = None
        self.d_pher_rel = None
        self.d_NN_idx = None
        self.d_NN_dist = None
        self.d_actb_score = None
        self.d_thresh = None
        self.d_pos_change = None

        if self.template.name == "predator":
            self.d_prey_detected = None
            self.d_prey_dir = None
        else:
            self.d_pred_detected = None
            self.d_pred_dir = None

        self.d_O = []  # organism state matrix  [Pop count, 1, N states]
        self.d_O_temp = []  # organism state matrix (temporary calculation)  [Pop count, 1, N states]
        self.d_O_temp_memory = []  # Memory node storage matrix
        self.d_O_temp_conduit = []
        self.d_W = []  # orgnaism weight matrix   [Pop count, N states, N states]
        self.d_B = []  # organism bias matrix  [Pop count, 1, N states]
        self.d_dropout_mask = [] # population node dropout mask (1 or 0) [Pop count, 1, # nodes] (0 means inactive node)

    def generate_random_population(self, initial_pos_sigma, sim_size=None, randomize_genes=True, position_count=None,
                                   distribution_type='Gaussian'):
        # Generate organisms
        org_rand_pos = initial_positions(no_org=self.pop_count, sigma=initial_pos_sigma,
                                         sim_size=sim_size, position_count=position_count,
                                         distribution_type=distribution_type)

        self.organisms = []
        for n in range(self.pop_count):
            #if organism_type == "numerical":
            self.organisms.append(
                Organism_Numerical(start_position=org_rand_pos[n, :], input_count=self.node_counts["input"],
                                   output_count=self.node_counts["output"],
                                   internal_neuron_count=self.node_counts["hidden"],
                                   conduit_neuron_count=self.node_counts["conduit"],
                                   memory_neuron_count=self.node_counts["memory"],
                                   max_connections=2000,
                                   gene_count=self.genes_per_organism,
                                   decision_thresh=self.decision_threshold))
        # randomize genes:
        if randomize_genes:
            for org in tqdm(self.organisms):
                org.randomize_genes()
                org.health = self.start_health

    def rand_movement_array(self, GPU=True):
        """
        Generate an array of random numbers for assigning to organism random movements
        :param GPU:
        :return:
        """
        rand_xy = np.random.rand(self.pop_count, 2)
        if not GPU:
            return rand_xy
        d_rand_xy = cuda.to_device(rand_xy)
        return d_rand_xy

    def reset_health(self, GPU=True):

        self.health = []
        for org in self.organisms:
            org.health = self.start_health
            self.health.append(org.health)
        self.health = np.array(self.health)

        if GPU:
            self.d_health = cuda.to_device(self.health)

    def update_mat(self, GPU=True):
        self.W = []
        self.B = []
        self.O = np.zeros((self.pop_count, 1, self.node_counts["total"]))  # organism state

        for org in self.organisms:
            conn_mat = org.brain.get_connections_mat()
            self.W.append(conn_mat[0])
            self.B.append(conn_mat[1])
        self.W = np.array(self.W)
        self.B = np.expand_dims(np.array(self.B), 1)

        if GPU:
            self.d_W = cuda.to_device(self.W)
            self.d_B = cuda.to_device(self.B)
            self.d_O = cuda.to_device(self.O)
            self.d_O_temp = cuda.device_array(self.O.shape, dtype='float', strides=None, order='C', stream=0)
            self.d_O_temp_memory = cuda.device_array(self.O.shape, dtype='float', strides=None, order='C', stream=0)
            self.d_O_temp_conduit = cuda.device_array(self.O.shape, dtype='float', strides=None, order='C', stream=0)

    def update_organism_arrays_cpu2gpu(self, GPU=True, sim_type="numerical"):

        self.pos = get_organism_pos(self.organisms)
        self.age = []
        self.clk_counter = []
        self.clk_lim = []
        self.clk_sig = []
        self.rr_counter = []
        self.rr_lim = []
        self.health = []
        self.alive = []
        self.mov_speed = []
        self.thresh = []
        self.pos_change = []
        for org in self.organisms:
            self.age.append(org.age)
            self.clk_counter.append(org.clk_counter)
            self.clk_lim.append(org.clk_limit)
            self.clk_sig.append(org.clk_sig)
            self.rr_counter.append(org.rr_counter)
            self.rr_lim.append(org.rr_limit)
            self.health.append(org.health)
            self.mov_speed.append(org.movement_speed)
            self.thresh.append(org.decision_threshold)
            self.pos_change.append(0.)
            if org.alive:
                self.alive.append(1)
            else:
                self.alive.append(0)
        if sim_type == "game":
            self.actb_score = []
            for org in self.organisms:
                self.actb_score.append(0)
            self.actb_score = np.array(self.actb_score)
        
        self.pos = np.array(self.pos)
        last_pos_array = np.zeros(self.pos.shape)
        last_pos_array[:, -1] = 1.
        self.pos_last = np.copy(last_pos_array)
        self.age = np.array(self.age)
        self.clk_counter = np.array(self.clk_counter)
        self.clk_lim = np.array(self.clk_lim)
        self.clk_sig = np.array(self.clk_sig)
        self.rr_counter = np.array(self.rr_counter)
        self.rr_lim = np.array(self.rr_lim)
        self.health = np.array(self.health)
        self.alive = np.array(self.alive)
        self.mov_speed = np.array(self.mov_speed)
        self.pher_rel = np.zeros(self.pop_count, dtype='float')
        self.thresh = np.array(self.thresh)
        self.pos_change = np.array(self.pos_change)

        if GPU:
            self.d_pos = cuda.to_device(self.pos)
            self.d_pos_last = cuda.to_device(self.pos_last)
            self.d_age = cuda.to_device(self.age)
            self.d_clk_counter = cuda.to_device(self.clk_counter)
            self.d_clk_lim = cuda.to_device(self.clk_lim)
            self.d_clk_sig = cuda.to_device(self.clk_sig)
            self.d_rr_counter = cuda.to_device(self.rr_counter)
            self.d_rr_lim = cuda.to_device(self.rr_lim)
            self.d_health = cuda.to_device(self.health)
            self.d_alive = cuda.to_device(self.alive)
            self.d_fitness = cuda.to_device(self.fitness)
            self.d_mv_speed = cuda.to_device(self.mov_speed)
            self.d_pher_rel = cuda.to_device(self.pher_rel)
            self.d_thresh = cuda.to_device(self.thresh)
            self.d_pos_change = cuda.to_device(self.pos_change)
            if self.template.name == "predator":
                self.d_prey_detected = cuda.device_array(self.pop_count, dtype='int', strides=None,
                                                         order='C', stream=0)
                self.d_prey_dir = cuda.device_array((self.pop_count, 3), dtype='float', strides=None, order='C',
                                                    stream=0)
            else:
                self.d_pred_detected = cuda.device_array(self.pop_count, dtype='int', strides=None,
                                                         order='C', stream=0)
                self.d_pred_dir = cuda.device_array((self.pop_count, 3), dtype='float', strides=None, order='C',
                                                    stream=0)
            if sim_type == "game":
                self.d_actb_score = cuda.to_device(self.actb_score)

    def update_organism_arrays_gpu2cpu(self):
        self.d_pos.copy_to_host(self.pos)
        self.d_alive.copy_to_host(self.alive)
        self.d_health.copy_to_host(self.health)
        self.d_fitness.copy_to_host(self.fitness)
        self.d_age.copy_to_host(self.age)
        self.d_thresh.copy_to_host(self.thresh)
        self.d_pos_change.copy_to_host(self.pos_change)
        if self.template.name == "predator":
            if self.prey_detected is None:
                self.prey_detected = np.zeros(self.d_prey_detected.shape, dtype='int')
                self.prey_dir = np.zeros(self.d_prey_dir.shape)
            self.d_prey_detected.copy_to_host(self.prey_detected)
            self.d_prey_dir.copy_to_host(self.prey_dir)

    def update_pos_last(self, USE_GPU=True):
        if USE_GPU:
            self.d_pos_last.copy_to_device(self.d_pos)
        else:
            self.pos_last = np.copy(self.pos)

    def set_debug_connections_custom(self, conn_list, bias, weight):
        from src.cortex import Connection
        debug_genes = []
        for i in range(len(conn_list)):
            debug_genes.append(Connection(src_node=conn_list[i][0], dst_node=conn_list[i][1], bias=bias, weight=weight))

        # fill out remaining genes
        for i in range(self.genes_per_organism - len(conn_list)):
            debug_genes.append(Connection(src_node=conn_list[0][0], dst_node=conn_list[0][1], bias=bias, weight=weight))

        self.mov_speed = 50

        for i in range(self.pop_count):
            self.organisms[i].brain.connections = []
            self.organisms[i].genes = []
            self.organisms[i].brain.connections += debug_genes
            self.organisms[i].genes += self.organisms[i].brain.get_connection_genes()

    def set_debug_connections_follow_grad(self, affected_organisms_count=None):

        from src.cortex import Connection
        outp_offs = self.input_count + self.internal_neuron_count
        debug_genes = []
        debug_genes.append(
            Connection(src_node=self.template.in_field_dx, dst_node=outp_offs + self.template.out_mov_rght, bias=-0.1,
                       weight=4))
        debug_genes.append(
            Connection(src_node=self.template.in_field_dx, dst_node=outp_offs + self.template.out_mov_lft, bias=-0.1,
                       weight=-4))
        debug_genes.append(
            Connection(src_node=self.template.in_field_dy, dst_node=outp_offs + self.template.out_mov_up, bias=-0.1,
                       weight=4))
        debug_genes.append(
            Connection(src_node=self.template.in_field_dy, dst_node=outp_offs + self.template.out_mov_dwn, bias=-0.1,
                       weight=-4))
        for i in range(self.genes_per_organism - 4):
            debug_genes.append(
                Connection(src_node=self.template.in_field_dy, dst_node=outp_offs + self.template.out_mov_up, bias=0,
                           weight=0))
        # for i in range(self.genes_per_organism - 4):
        #    debug_genes.append(Connection(src_node=in_field_dy, dst_node=out_mov_dwn, bias=-0.1, weight=-4))
        if affected_organisms_count is None:
            affected_organisms_count = self.pop_count
        for i in range(affected_organisms_count):
            self.organisms[i].brain.connections = []
            self.organisms[i].genes = []
            self.organisms[i].brain.connections += debug_genes
            self.organisms[i].genes += self.organisms[i].brain.get_connection_genes()

    def set_debug_connections_numerical_crypto(self, affected_organisms_count=None):

        from src.cortex import Connection
        outp_offs = self.input_count + self.internal_neuron_count
        debug_genes = []
        debug_genes.append(
            Connection(src_node=self.template.static_input_count, dst_node=outp_offs + self.template.out_btc, bias=0,
                       weight=4))
        debug_genes.append(
            Connection(src_node=self.template.static_input_count + 1, dst_node=outp_offs + self.template.out_eth,
                       bias=0, weight=4))
        debug_genes.append(
            Connection(src_node=self.template.static_input_count + 2, dst_node=outp_offs + self.template.out_ltc,
                       bias=0, weight=4))
        for i in range(self.genes_per_organism - 4):
            debug_genes.append(Connection(src_node=0, dst_node=outp_offs, bias=0, weight=0))
        # for i in range(self.genes_per_organism - 4):
        #    debug_genes.append(Connection(src_node=in_field_dy, dst_node=out_mov_dwn, bias=-0.1, weight=-4))
        if affected_organisms_count is None:
            affected_organisms_count = self.pop_count
        for i in range(affected_organisms_count):
            self.organisms[i].brain.connections = []
            self.organisms[i].genes = []
            self.organisms[i].brain.connections += debug_genes
            self.organisms[i].genes += self.organisms[i].brain.get_connection_genes()

    def calc_decision_GPU(self, sim_size, prey_population=None, sim_type="2D", conduit_solve_depth=1):
        """
        Node order: [input, conduit, hidden, memory, output]
        :param sim_size:
        :param prey_population:
        :param sim_type:
        :param conduit_solve_depth:
        :return:
        """
        threadsperblock_matmul = (32, 32)
        blockspergrid_x = math.ceil(self.node_counts["total"] / threadsperblock_matmul[0])  # matrix width
        blockspergrid_y = math.ceil(self.pop_count / threadsperblock_matmul[1])  # batch
        blockspergrid_matmul = (blockspergrid_x, blockspergrid_y)

        blockspergrid_x = math.ceil(self.pop_count / threadsperblock_matmul[0])  # batch
        blockspergrid_y = math.ceil(self.node_counts["total"] / threadsperblock_matmul[1])  # matrix width
        blockspergrid_copy = (blockspergrid_x, blockspergrid_y)

        #move this gpu array allocation to init
        #_Copy_2DArray[blockspergrid_copy, threadsperblock_matmul](self.d_O, self.d_O_temp, self.node_counts["input"],
        #                                                          self.node_counts["input"] + self.node_counts["conduit"])
        #_Zero_and_Copy_2DArray[blockspergrid_copy, threadsperblock_matmul](self.d_O, self.d_O_temp, self.node_counts["input"],
        #                                                          self.node_counts["input"] + self.node_counts["conduit"])
        # For conduit node calculation, copy over all nodes from O->O_temp except for the existing conduit node values
        # and the output node values
        _zero_and_copy_2d_array[blockspergrid_copy, threadsperblock_matmul](self.d_O, self.d_O_temp, 0,
                                                                           self.node_counts["input"])
        _copy_2d_array[blockspergrid_copy, threadsperblock_matmul](self.d_O, self.d_O_temp, self.node_counts["input"] + self.node_counts["conduit"],
                                                                  self.node_counts["total"] - self.node_counts["output"])

        # Save starting memory node values (for resetting when 'store' not activated)
        if self.node_counts["memory"] > 0:
            _copy_2d_array[blockspergrid_copy, threadsperblock_matmul](self.d_O, self.d_O_temp_memory,
                                                                      self.node_counts["input"] + self.node_counts["conduit"] + self.node_counts["hidden"],
                                                                      self.node_counts["total"] - self.node_counts["output"])

        for ic in range(conduit_solve_depth):
            # Compute new conduit neuron outputs. Only updating conduit neuron values
            _matmul[blockspergrid_matmul, threadsperblock_matmul](self.d_O_temp, self.d_W, self.d_B, self.d_O_temp_conduit)
            # Flatten conduit nodes
            flatten_cuda_array(self.d_O_temp_conduit, self.d_pos.shape[1])
            # Zero all non-conduit neurons (not necessary, we just only copy over the conduit node values)
            #self.d_O_temp_conduit[:, :, :self.input_count] = 0
            #self.d_O_temp_conduit[:, :, (self.input_count + self.conduit_neuron_count):] = 0
            # _Copy_2DArray(array_src, array_dst, start_idx, end_idx)
            _copy_2d_array[blockspergrid_copy, threadsperblock_matmul](self.d_O_temp_conduit, self.d_O_temp, self.node_counts["input"],
                                                                      self.node_counts["input"] + self.node_counts["conduit"])

            _copy_2d_array[blockspergrid_copy, threadsperblock_matmul](self.d_O_temp, self.d_O, self.node_counts["input"],
                                                                  self.node_counts["input"] + self.node_counts["conduit"])

        # Matrix multiply d_O with d_W and add d_B to get d_O_temp
        _matmul[blockspergrid_matmul, threadsperblock_matmul](self.d_O, self.d_W, self.d_B, self.d_O_temp)

        # Copy array values from d_O_temp to d_O (except for input nodes)
        _copy_2d_array[blockspergrid_copy, threadsperblock_matmul](self.d_O_temp, self.d_O, self.node_counts["input"], self.node_counts["total"])

        # Apply flattening function to node outputs (eg. tanh)
        flatten_cuda_array(self.d_O, self.d_pos.shape[1])

        # If there are population nodes, average their values across the population and replace the node values for each
        # organism
        if self.population_node_count > 0:
            pop_node_start = self.node_counts["input"] + self.node_counts["conduit"] + self.node_counts["hidden"] - self.population_node_count
            pop_node_end = self.node_counts["input"] + self.node_counts["conduit"] + self.node_counts["hidden"]
            threadsperblock_pop_nodes = 256
            blockspergrid_pop_nodes = (self.node_counts["total"] + threadsperblock_pop_nodes - 1) // threadsperblock_pop_nodes
            _average_broadcast[blockspergrid_pop_nodes, threadsperblock_pop_nodes](self.d_O, pop_node_start, pop_node_end)

        # Reset memory nodes of non-activated memory store signals
        if self.node_counts["memory"] > 0:
            # src array: d_O_temp_memory, dest array: d_O, gate control signals: d_O
            # _Gated_Copy_2DArray(array_src, array_dst, start_idx, end_idx, gate_start_idx, gate_store_threshold)
            # array_dst[x, 0, gate_no + gate_start_idx] > gate_store_threshold
            # y = 272
            # gate_no = y - start_idx = 0
            # gate_no + gate_start_idx = 0 + 290 = 290
            _gated_copy_2d_array[blockspergrid_copy, threadsperblock_matmul](self.d_O_temp_memory, self.d_O,
                                                                      self.node_counts["input"] + self.node_counts["conduit"] + self.node_counts["hidden"],
                                                                      self.node_counts["total"] - self.node_counts["output"],
                                                                      self.out_offset + self.template.out_idx["MEM store 0"],
                                                                      self.decision_threshold,
                                                                      )

        # Apply dropout
        if len(self.dropout_mask) > 0:
            _self_matrix_elementwise_multiply[blockspergrid_copy, threadsperblock_matmul](self.d_O, self.d_dropout_mask)

        # Determine output behaviours based on node outputs
        calc_decide(population=self, sim_size=sim_size, prey_population=prey_population, sim_type=sim_type)

        if self.debug_target_org is not None:
            self.debug_output_states = np.zeros(self.O.shape)
            self.d_O.copy_to_host(self.debug_output_states)
            self.debug_output_states = self.debug_output_states[self.debug_target_org, 0, :]

    def calc_decision_CPU(self, sim_size, prey_population=None, sim_type="2D'", conduit_solve_depth=3):


        # Save starting memory node values (for resetting when 'store' not activated)
        O_temp_memory = None
        if self.node_counts["memory"] > 0:
            O_temp_memory = self.O.copy()

        # Compute the new conduit neuron outputs
        # For conduit calculation inputs in O_temp, copy over all nodes from O except for the previous conduit node
        # values and the output node values
        O_temp = np.zeros(self.O.shape)
        O_temp[:, :, 0:self.node_counts["input"]] = self.O.copy()[:, :, 0:self.node_counts["input"]]
        O_temp[:, :, (self.node_counts["input"]+self.node_counts["conduit"]):(self.node_counts["total"] - self.node_counts["output"])] =\
            self.O.copy()[:, :, (self.node_counts["input"]+self.node_counts["conduit"]):(self.node_counts["total"] - self.node_counts["output"])]
        #O_temp = self.O.copy()
        for ic in range(conduit_solve_depth):
            O_temp_new = np.zeros(self.O.shape)
            matmul(O_temp, self.W, self.B, O_temp_new)
            # Flatten conduit nodes
            flatten(O_temp_new, self.pos.shape[1])
            # Zero all non-conduit neurons
            O_temp_new[:, :, :self.node_counts["input"]] = 0
            O_temp_new[:, :, (self.node_counts["input"] + self.node_counts["conduit"]):] = 0
            O_temp = O_temp_new.copy()

        # Copy only conduit neuron values to starting node states
        self.O[:, :, self.node_counts["input"]:(self.node_counts["input"] + self.node_counts["conduit"])] = O_temp[:, :, self.node_counts["input"]:(self.node_counts["input"] + self.node_counts["conduit"])].copy()

        # Calculate all remaining nodes
        O_temp = np.zeros(self.O.shape)
        matmul(self.O, self.W, self.B, O_temp)

        # Copy array values from O_temp to O (except for input nodes)
        self.O[:, :, self.node_counts["input"]:] = O_temp[:, :, self.node_counts["input"]:].copy()

        # Apply flattening function to node outputs (eg. tanh)
        flatten(self.O, self.pos.shape[1])

        # If there are population nodes, average their values across the population and replace the node values for each
        # organism
        if self.population_node_count > 0:
            pop_node_start = self.node_counts["input"] + self.node_counts["conduit"] + self.node_counts["hidden"] - self.population_node_count
            pop_node_end = self.node_counts["input"] + self.node_counts["conduit"] + self.node_counts["hidden"]
            self.O[:, 0, pop_node_start:pop_node_end] = np.tile(self.O[:, 0, pop_node_start:pop_node_end].mean(0), (self.O.shape[0], 1))

        # Reset memory nodes of non-activated memory store signals
        if self.node_counts["memory"] > 0:
            mem_nodes_start = self.node_counts["input"] + self.node_counts["conduit"] + self.node_counts["hidden"]
            mem_nodes_end = self.node_counts["total"] - self.node_counts["output"]
            mem_store_start = self.out_offset + self.template.out_idx["MEM store 0"]
            mem_store_end = mem_store_start + self.node_counts["memory"]
            revert_mask = np.zeros(self.O.shape)
            revert_mask[:, 0, mem_nodes_start:mem_nodes_end] = self.O[:, 0, mem_store_start:mem_store_end] < self.decision_threshold
            revert_mask = revert_mask == 1.
            self.O[revert_mask] = O_temp_memory[revert_mask]


        # Apply dropout
        if len(self.dropout_mask) > 0:
            self.O *= self.dropout_mask

        # Determine output behaviours based on node outputs
        calc_decide(population=self, sim_size=sim_size, prey_population=prey_population, sim_type=sim_type,
                    USE_GPU=False)

    def apply_position_mask(self, env, USE_GPU=True):
        if env.position_mask is not None:
            apply_mask(pos=self.d_pos if USE_GPU else self.pos,
                       pos_mask=env.d_position_mask if USE_GPU else env.position_mask,
                       USE_GPU=USE_GPU)


    def calc_NN_GPU(self, copy_gpu2cpu=False):
        """
        Calculate nearest neighbour index and distance for each organism in a populatiojn
        :param copy_gpu2cpu:
        :return:
        """

        self.d_NN_idx = cuda.device_array(self.pop_count, dtype='int', strides=None, order='C', stream=0)
        self.d_NN_dist = cuda.device_array(self.pop_count, dtype='float', strides=None, order='C', stream=0)

        threadsperblock = 16
        blockspergrid = math.ceil(self.pop_count / threadsperblock)
        _find_closest_2d[blockspergrid, threadsperblock](self.d_pos, self.d_NN_idx, self.d_NN_dist)

        if copy_gpu2cpu:
            self.NN_idx = np.empty(self.pop_count, dtype='int')
            self.NN_dist = np.empty(self.pop_count)
            self.d_NN_idx.copy_to_host(self.NN_idx)
            self.d_NN_dist.copy_to_host(self.NN_dist)

    def calc_find_prey(self, prey_population, sensory_radius, copy_gpu2cpu=False):

        self.d_prey_dist = cuda.device_array(self.pop_count, dtype='float', strides=None, order='C', stream=0)
        self.d_prey_dir_x = cuda.device_array(self.pop_count, dtype='float', strides=None, order='C', stream=0)
        self.d_prey_dir_y = cuda.device_array(self.pop_count, dtype='float', strides=None, order='C', stream=0)
        d_closest_idx = cuda.device_array(self.pop_count, dtype='int', strides=None, order='C', stream=0)

        threadsperblock = 16
        blockspergrid = math.ceil(self.pop_count / threadsperblock)
        _find_closest_2d_species[blockspergrid, threadsperblock](self.d_pos, prey_population.d_pos,
                                                               d_closest_idx, self.d_prey_dist)

        if copy_gpu2cpu:
            self.prey_dist = np.empty(self.pop_count)
            self.d_prey_dist.copy_to_host(self.prey_dist)

    def generate_dropout_mask(self, dropout_rate, USE_GPU=False):
        self.dropout_mask = np.array((np.random.rand(self.pop_count, 1, self.node_counts['total']) > dropout_rate)).astype('float')
        self.d_dropout_mask = cuda.to_device(self.dropout_mask)

    def parameter_summary(self, print_str=False):
        out_str = "{:-^70}\n".format("Population \'{}\' Parameters".format(self.pop_label))
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Parameter", "Value")
        out_str += 70 * "-" + "\n"
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Population Size", self.pop_count)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Node Counts", "")
        for k in self.node_counts.keys():
            out_str += "|\t     {:<25}|\t\t{:<30}|\n".format(k, self.node_counts[k])
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Population Nodes", self.population_node_count)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Genes/Organism", self.genes_per_organism)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Max Connection Count", self.max_connections)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Decision Threshold (start)", self.decision_threshold)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Spread Enabled", self.use_spread)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Transaction Fee Enabled", self.use_trans_fee)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Fitness Type", self.fitness_type)
        out_str += "|\t{:<30}|\t\t{:<30}|\n".format("Fitness Cap", self.fitness_cap if self.fitness_cap else "None")
        out_str += "-" * 70 + "\n"
        if print_str:
            print(out_str)
        return out_str

class PopulationConfig:
    def __init__(self, pop_count=None, genes_per_organism=None,
                 internal_neuron_count=None, conduit_neuron_count=None, memory_neuron_count=None,
                 population_node_count=0,
                 decision_threshold=None,
                 max_health=10000., min_health=-10000., start_health=50.,
                 initial_pos_sigma=None,
                 fitness_type='vector-health',
                 max_mov_speed=10, max_CLK_period=35, max_RR_period=50,
                 max_connections=None,
                 pher_channel=None,
                 PD_channel=None,
                 vision_radius=None, eat_radius=None, use_spread=False, use_trans_fee=False,
                 pop_label="", fan_color=(0, 0, 1), line_color=(0, .3, 1),
                 config_dict=None, fitness_cap=None):
        super(PopulationConfig, self).__init__()

        self.max_health = max_health
        self.min_health = min_health
        self.start_health = start_health
        self.vision_radius = vision_radius
        self.max_mov_speed = max_mov_speed
        self.initial_pos_sigma = initial_pos_sigma
        self.fitness_cap = fitness_cap

        if config_dict is not None:
            self.pop_count = config_dict["population count"]
            self.genes_per_organism = config_dict["genes per organism"]
            self.internal_neuron_count = config_dict["internal neuron count"]
            self.conduit_neuron_count = config_dict["conduit neuron count"]
            self.memory_neuron_count = config_dict["memory neuron count"]
            self.population_node_count = config_dict["population node count"]
            self.decision_threshold = config_dict["decision threshold"]
            self.fitness_type = config_dict["fitness type"]
            self.max_CLK_period = config_dict["max CLK period"]
            self.max_RR_period = config_dict["max RR period"]
            self.max_connections = config_dict["max connections"]
            self.pher_channel = config_dict['pher channel']
            self.use_spread = config_dict['use spread']
            self.use_trans_fee = config_dict['use transaction fee']
            self.PD_channel = config_dict['PD channel']
            self.pop_label = config_dict['pop label']
            self.fan_color = np.array(config_dict['fan color'])
            self.line_color = np.array(config_dict['line color'])
        elif pop_count is not None:
            self.pop_count = pop_count  # number of organisms
            self.genes_per_organism = genes_per_organism

            self.max_CLK_period = max_CLK_period
            self.max_RR_period = max_RR_period
            self.internal_neuron_count = internal_neuron_count
            self.conduit_neuron_count = conduit_neuron_count
            self.memory_neuron_count = memory_neuron_count
            self.population_node_count = population_node_count
            self.fitness_type = fitness_type

            self.decision_threshold = decision_threshold  # should replace this with softmax for related/conflicting decisions
            self.eat_radius = eat_radius
            self.use_spread = use_spread  # if True, use ask/bid spread (equivalent fee) in ROI calculation
            self.use_trans_fee = use_trans_fee
            self.pher_channel = pher_channel
            self.PD_channel = PD_channel
            self.pop_label = pop_label
            self.fan_color = np.array(fan_color)
            self.line_color = np.array(line_color)
        else:
            print("ERROR: Invalid PopulationConfig parameters.")
            return

        if max_connections is None:
            self.max_connections = self.genes_per_organism
        else:
            self.max_connections = max_connections

        if self.genes_per_organism < 10:
            print("Warning: using only {} genes/organism".format(self.genes_per_organism))


class OrganismDebugProbe:
    def __init__(self, population_idx, organism_idx):
        super(OrganismDebugProbe, self).__init__()
        self.population_idx = population_idx
        self.organism_idx = organism_idx
        self.W = None
        self.B = None

        self.logged_states = []
        self.logged_positions = []

        self.node_counts = None
        self.static_input_names = None
        self.data_input_names = None
        self.position_density_input_names = None

    def log_parameters(self, populations, GPU=False, environment=None):

        if GPU:
            self.W = populations[self.population_idx].d_W.copy_to_host()[self.organism_idx, :, :]
            self.B = populations[self.population_idx].d_B.copy_to_host()[self.organism_idx, 0, :]
        else:
            self.W = np.copy(populations[self.population_idx].W[self.organism_idx, :, :])
            self.B = np.copy(populations[self.population_idx].B[self.organism_idx, 0, :])
        self.node_counts = populations[self.population_idx].node_counts
        self.static_input_names = populations[self.population_idx].template.input_names
        if environment is not None:
            self.data_input_names = environment.data_in_names
            self.position_density_input_names = environment.position_density_names

    def log_state(self, populations, stage_label, t, GPU=False):

        if len(self.logged_states) < t+1:
            self.logged_states.append({"t": t})
        if len(self.logged_positions) < t+1:
            self.logged_positions.append({"t": t})

        if GPU:
            self.logged_states[t][stage_label] = populations[self.population_idx].d_O.copy_to_host()[self.organism_idx, 0, :]
            self.logged_positions[t][stage_label] = populations[self.population_idx].d_pos.copy_to_host()[self.organism_idx, :]
        else:
            self.logged_states[t][stage_label] = np.copy(populations[self.population_idx].O[self.organism_idx, 0, :])
            self.logged_positions[t][stage_label] = np.copy(populations[self.population_idx].pos[self.organism_idx, :])

    def save(self, path):
        import pickle
        pickle.dump(self, open(path, "wb"))

    def cross_compare(self, other_probe, self_label, other_label, max_t=None):

        stop_t = len(self.logged_states)
        if max_t:
            stop_t = max_t

        # Compare weights
        print(20*"*" + " Comparing OrganismProbe (Population {}, Organism {}) {} - {}".format(self.population_idx, self.organism_idx, self_label, other_label) + 20*"*")
        if not np.array_equal(self.W, other_probe.W):
            print("Weight matrix mismatch")
            print("{} Weight matrix nonzero elements:".format(self_label))
            print(np.where(np.abs(self.W) > 0.01))
            print("{} Weight matrix nonzero elements:".format(other_label))
            print(np.where(np.abs(other_probe.W) > 0.01))
            return False
        else:
            print("Weight matrix match")
        # Compare biases
        if not np.array_equal(self.B, other_probe.B):
            print("Bias vector mismatch")
            return False
        else:
            print("Bias vector match")

        print(10*"-" + " Comparing States/Positions " + 10*"-")
        first_state_mistmach_t = -1
        first_state_mistmach_label = -1
        print("_"*50)
        print("|\tt\t|\tObject\t|\tStage\t|\tMatch\t|")
        print("-"*50)
        for t in range(stop_t):
            for k in range(len(self.logged_states[t].keys())-1):
                state_name = list(self.logged_states[t].keys())[k+1]
                states_diff = np.abs(self.logged_states[t][state_name] - other_probe.logged_states[t][state_name])
                if states_diff.max() > 1e-5:
                    arrays_match = False
                else:
                    arrays_match = True
                #arrays_match = np.array_equal(self.logged_states[t][state_name],
                #                              other_probe.logged_states[t][state_name])
                print("\t{}\t\t{}\t\t{}\t\t{}".format(t, "state", state_name, arrays_match))
                if not arrays_match and first_state_mistmach_t==-1:
                    first_state_mistmach_t = t
                    first_state_mistmach_label = state_name

        print("_"*50)
        print("|\tt\t|\tObject\t|\tStage\t|\tMatch\t|")
        print("-"*50)
        for t in range(stop_t):
            for k in range(len(self.logged_positions[t].keys())-1):
                state_name = list(self.logged_positions[t].keys())[k+1]
                arrays_match = np.array_equal(self.logged_positions[t][state_name],
                                              other_probe.logged_positions[t][state_name])
                print("\t{}\t\t{}\t\t{}\t\t{}".format(t, "positions", state_name, arrays_match))

        if first_state_mistmach_t > -1:
            print("\n" + 10 * "-" + " State Mismatch Analysis " + 10 * "-")
            node_types = []
            for i in range(self.node_counts['total']):
                if i < self.node_counts['input']:
                    if self.data_input_names is None:
                        node_types.append("input {}".format(i))
                    else:
                        if i < len(self.static_input_names):
                            node_types.append("{}(in{})".format(self.static_input_names[i], i))
                        elif i < len(self.static_input_names) + len(self.data_input_names):
                            node_types.append("{}(in{})".format(self.data_input_names[i - len(self.static_input_names)], i))
                        else:
                            node_types.append("{}(in{})".format(self.position_density_input_names[i - len(self.static_input_names) - len(self.data_input_names)], i))
                elif i < (self.node_counts['input'] + self.node_counts['conduit']):
                    node_types.append("conduit {}".format(i-self.node_counts['input']))
                elif i < (self.node_counts['input'] + self.node_counts['conduit'] + self.node_counts['hidden']):
                    node_types.append("hidden {}".format(i - self.node_counts['input'] - self.node_counts['conduit']))
                elif i < (self.node_counts['input'] + self.node_counts['conduit'] + self.node_counts['hidden'] + self.node_counts['memory']):
                    node_types.append("memory {}".format(i - self.node_counts['input'] - self.node_counts['conduit'] - self.node_counts['hidden']))
                elif i < (self.node_counts['input'] + self.node_counts['conduit'] + self.node_counts['hidden'] + self.node_counts['memory'] + self.node_counts['output']):
                    node_types.append("output {}".format(i - self.node_counts['input'] - self.node_counts['conduit'] - self.node_counts['hidden'] - self.node_counts['memory']))

            #for i in range(len(node_types)):
            #    print(i, node_types[i])
            import pickle
            pickle.dump(node_types, open("GPU_CPU_compare/state_node_names.p", "wb"))
            states1 = self.logged_states[first_state_mistmach_t][first_state_mistmach_label]
            states2 = other_probe.logged_states[first_state_mistmach_t][first_state_mistmach_label]
            states_diff = np.abs(states2-states1)
            states_mismatch_idx = np.where(states_diff > 1e-5)
            print("Mismatched nodes:")
            print(np.array(node_types)[states_mismatch_idx])