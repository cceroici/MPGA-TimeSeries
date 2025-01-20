import numpy as np
import random

'''
Gene types:

total: 64 bits
or 
16 hex digits

0 - connection
1 - internal clock starting period
2 - movement speed starting value
3 - reaction rate starting period
4 - random connection path
5 - source influence connection (backward)
6 - source influence connection (forward)
7 - reinforcement connection
8 - decision threshold starting value
15 - latent gene (no effect)

--connected gene format--
[gene type] [crossover mult] [src node] [dst node] [weight] [ bias ]
[ 4-bit   ] [     4-bit    ] [ 12-bit ] [ 12-bit ] [16-bit] [16-bit]
[ 1-hex   ] [     1-hex    ] [ 3-hex  ] [ 3-hex  ] [4-hex ] [4-hex ]

--internal timer period (CLK INIT) format--
[gene type] [crossover mult] [ period ] [          empty           ]
[ 4-bit   ] [     4-bit    ] [ 8-bit  ] [          48-bit          ]
[ 1-hex   ] [     1-hex    ] [ 2-hex  ] [          12-hex          ]

--movement speed format--
[gene type] [crossover mult] [ period ] [          empty           ]
[ 4-bit   ] [     4-bit    ] [ 8-bit  ] [          48-bit          ]
[ 1-hex   ] [     1-hex    ] [ 2-hex  ] [          12-hex          ]

--reaction rate period format--
[gene type] [crossover mult] [ period ] [          empty           ]
[ 4-bit   ] [     4-bit    ] [ 8-bit  ] [          48-bit          ]
[ 1-hex   ] [     1-hex    ] [ 2-hex  ] [          12-hex          ]

--decision threshold (dec thresh INIT) format--
[gene type] [crossover mult] [ thresh ] [          empty           ]
[ 4-bit   ] [     4-bit    ] [ 8-bit  ] [          48-bit          ]
[ 1-hex   ] [     1-hex    ] [ 2-hex  ] [          12-hex          ]

--random connection path gene format--
[gene type] [crossover mult] [src node] [dst node] [weight] [# of intermediate]
[ 4-bit   ] [     4-bit    ] [ 12-bit ] [ 12-bit ] [16-bit] [      16-bit     ]
[ 1-hex   ] [     1-hex    ] [ 3-hex  ] [ 3-hex  ] [4-hex ] [      4-hex      ]

--source influence (forward) gene format--
[gene type] [crossover mult] [src node] [dst node] [weight] [ None ]
[ 4-bit   ] [     4-bit    ] [ 12-bit ] [ 12-bit ] [16-bit] [16-bit]
[ 1-hex   ] [     1-hex    ] [ 3-hex  ] [ 3-hex  ] [4-hex ] [4-hex ]

--source influence (backward) gene format--
[gene type] [crossover mult] [src node] [dst node] [weight] [ None ]
[ 4-bit   ] [     4-bit    ] [ 12-bit ] [ 12-bit ] [16-bit] [16-bit]
[ 1-hex   ] [     1-hex    ] [ 3-hex  ] [ 3-hex  ] [4-hex ] [4-hex ]

--reinforcement connection gene format--
[gene type] [crossover mult] [src node] [dst node] [weight] [ bias ]
[ 4-bit   ] [     4-bit    ] [ 12-bit ] [ 12-bit ] [16-bit] [16-bit]
[ 1-hex   ] [     1-hex    ] [ 3-hex  ] [ 3-hex  ] [4-hex ] [4-hex ]

--latent gene format--
[gene type] [crossover mult] [               None                  ]
[ 4-bit   ] [     4-bit    ] [              56-bit                 ]
[ 1-hex   ] [     1-hex    ] [              14-hex                 ]
'''

# Size (in bits) of the source/destination node encodings
src_node_bits = 12
dst_node_bits = 12

# maximum magnitude of possible weights/biases
max_weight = 8
max_bias = 4

# maximum number of intermediate connections in random path connections genes
rpath_max_im = 3

# maximum/minimum starting movement speed
max_mvmt_speed = 5
min_mvmt_speed = 1

# maximum/minimum starting CLK period
base_max_CLK_lim = 50
base_min_CLK_lim = 1

# maximum/minimum starting reaction rate
base_max_RR_lim = 50
base_min_RR_lim = 1

# maximum/minimum starting decision threshold
base_max_dec_thresh = 0.98
base_min_dec_thresh = 0.2

# maximum magnitude of crossover probability multiplier
max_crossover_mult = 2.

gene_codes = ['0', '1', '2', '3', '4', '5', '6', '7', 'f', '8']
gene_type_labels = ['direct-conn', 'clk-init', 'mvmt-speed', 'reaction-rate', 'random-path-conn',
                    'src-influence-conn', 'dst-influence-conn', 'reinforcement-conn', 'latent', 'decision-thresh']


def chunk(in_string, num_chunks):
    chunk_size = len(in_string) // num_chunks
    if len(in_string) % num_chunks: chunk_size += 1
    iterator = iter(in_string)
    for _ in range(num_chunks):
        accumulator = list()
        for _ in range(chunk_size):
            try:
                accumulator.append(next(iterator))
            except StopIteration:
                break
        yield ''.join(accumulator)


def random_gene(node_counts, type="numerical"):
    r = np.random.rand()
    crosvr_mlt = np.random.rand() * (1.3 - 0.7) + 0.7
    #crosvr_mlt = np.random.rand() * 2.
    #crosvr_mlt = 1.
    if type == "2D":
        if r < 0.92:
            return generate_random_connection_gene(node_counts, crosvr_mlt)
        elif r < 0.94:
            return generate_random_movement_speed_gene(crosvr_mlt)
        else:
            return generate_random_clk_init_gene(crosvr_mlt)
    elif type == "numerical":
        if r < 0.65:
            r_conn = np.random.rand()
            #r_conn = 0.1
            if r_conn < 0.75:
                conn_type = "single"
            elif r_conn < 0.8:
                conn_type = "random path"
            elif r_conn < 0.87:
                conn_type = "influence-bwd"
            elif r_conn < 0.93:
                conn_type = "influence-fwd"
            else:
                conn_type = "reinforcement"
            return generate_random_connection_gene(node_counts=node_counts, conn_type=conn_type, crosvr_mlt=crosvr_mlt)
        elif r < 0.70:
            return generate_random_clk_init_gene(crosvr_mlt)
        elif r < 0.75:
            return generate_random_rr_init_gene(crosvr_mlt)
        elif r < 0.8:
            return generate_random_decision_threshold_init_gene(crosvr_mlt)
        else:
            return latent_gene(crosvr_mlt)


def generate_random_clk_init_gene(crosvr_mlt=0.5):
    timer_period = int(np.round(np.random.rand() * (base_max_CLK_lim - base_min_CLK_lim) + base_min_CLK_lim))
    code = encode_clk_init_gene(timer_period=timer_period, crosvr_mlt=crosvr_mlt)
    return code


def generate_random_decision_threshold_init_gene(crosvr_mlt=0.5):
    decision_threshold = np.random.rand() * (base_max_dec_thresh - base_min_dec_thresh) + base_min_dec_thresh
    code = encode_decision_threshold_init_gene(dec_thresh=decision_threshold, crosvr_mlt=crosvr_mlt)
    return code


def generate_random_rr_init_gene(crosvr_mlt=0.5):
    RR_period = int(np.round(np.random.rand() * (base_max_RR_lim - base_min_RR_lim) + base_min_RR_lim))
    code = encode_rr_init_gene(rr_period=RR_period, crosvr_mlt=crosvr_mlt)
    return code


def latent_gene(crosvr_mlt=0.5):
    gene_code = "{0:#0{1}x}".format(int(15), 1)[2:]
    gene_code += "{0:#0{1}x}".format(int(np.round(crosvr_mlt / max_crossover_mult * 15)), 1)[2:]  # crossover multiplier
    gene_code += 12 * "0"
    return gene_code


def generate_random_movement_speed_gene(crosvr_mlt=0.5):
    mov_speed = int(np.round(np.random.rand() * (max_mvmt_speed - min_mvmt_speed) + min_mvmt_speed))
    code = encode_movement_speed_gene(mov_speed=mov_speed, crosvr_mlt=crosvr_mlt)
    return code


def generate_random_connection_gene(node_counts, conn_type="single", crosvr_mlt=0.5):

    """
    Possible connections:
    1. input - cNeuron
    2. hidden - cNeuron
    3. cNeuron - hidden
    4. cNeuron - Output
    :param input_count:
    :param output_count:
    :param internal_neuron_count:
    :param conduit_neuron_count:
    :param crosvr_mlt:
    :return:
    """
    src_node = int(np.round(np.random.rand() * (node_counts['input'] + node_counts['hidden'] + node_counts['conduit'] - 1)))
    if src_node > node_counts['input'] and src_node < (node_counts['input'] + node_counts['conduit']):  # src_node is conduit neuron
        dst_node = int(np.round(np.random.rand() * (node_counts['hidden'] + node_counts['output'] - 1) + node_counts['input'] + node_counts['conduit']))
    else:
        if np.random.rand() > 0.5:  # destination: output node
            dst_node = int(np.round(np.random.rand() * (node_counts['output'] - 1) + node_counts['input'] + node_counts['conduit'] + node_counts['hidden']))
        else:  # destination: internal node
            dst_node = int(np.round(np.random.rand() * (node_counts['hidden'] + node_counts['conduit'] - 1) + node_counts['input']))

    weight = (np.random.rand() * 2 - 1) * max_weight
    bias = (np.random.rand() * 2 - 1) * max_bias

    if src_node >= (2**src_node_bits-1):
        print("ERROR (Genome.py:generate_random_connection_gene(): source node ({})"
              " exceeds max gene bit encoding ({})".format(src_node, (2**src_node_bits-1)))
        return None
    elif dst_node >= (2**dst_node_bits-1):
        print("ERROR (Genome.py:generate_random_connection_gene(): destination node ({})"
              " exceeds max gene bit encoding ({})".format(dst_node, (2**dst_node_bits-1)))
        return None

    if conn_type == "single":
        code = encode_connection_gene(src_node=src_node, dst_node=dst_node, weight=weight, bias=bias, crosvr_mlt=crosvr_mlt)
    elif conn_type == "random path":
        no_intermediate = np.random.randint(1, rpath_max_im)
        code = encode_random_path_connection_gene(src_node=src_node, dst_node=dst_node, weight=weight,
                                                  no_intermediate=no_intermediate, crosvr_mlt=crosvr_mlt)
    elif conn_type == "influence-bwd":
        code = encode_influence_connection_gene(src_node=src_node, dst_node=dst_node, weight=weight,
                                                crosvr_mlt=crosvr_mlt, backward=True)
    elif conn_type == "influence-fwd":
        code = encode_influence_connection_gene(src_node=src_node, dst_node=dst_node, weight=weight,
                                                crosvr_mlt=crosvr_mlt, backward=False)
    elif conn_type == "reinforcement":
        code = encode_reinforcement_connection_gene(src_node=src_node, dst_node=dst_node, weight=weight, bias=bias,
                                                    crosvr_mlt=crosvr_mlt)
    else:
        print("WARNING - Generate_Random_Connection_Gene(): Invalid connection gene type \"{}\"".format(conn_type))
        return None

    return code


def encode_clk_init_gene(timer_period, crosvr_mlt=0.5):
    if timer_period > base_max_CLK_lim:
        timer_period = base_max_CLK_lim
    gene_code = "1"
    gene_code += "{0:#0{1}x}".format(int(np.round(crosvr_mlt / max_crossover_mult * 15)), 1)[2:]  # crossover multiplier
    gene_code += "{0:#0{1}x}".format(timer_period, 4)[2:]
    gene_code += 12 * "0"
    return gene_code


def encode_decision_threshold_init_gene(dec_thresh, crosvr_mlt=0.5):
    if dec_thresh > base_max_dec_thresh:
        dec_thresh = base_max_dec_thresh
    gene_code = "8"
    gene_code += "{0:#0{1}x}".format(int(np.round(crosvr_mlt / max_crossover_mult * 15)), 1)[2:]  # crossover multiplier
    gene_code += "{0:#0{1}x}".format(int(dec_thresh*255), 4)[2:]
    gene_code += 12 * "0"
    return gene_code


def encode_rr_init_gene(rr_period, crosvr_mlt=0.5):
    if rr_period > base_max_RR_lim:
        rr_period = base_max_RR_lim
    elif rr_period < base_min_RR_lim:
        rr_period = base_min_RR_lim

    gene_code = "3"
    gene_code += "{0:#0{1}x}".format(int(np.round(crosvr_mlt / max_crossover_mult * 15)), 1)[2:]  # crossover multiplier
    gene_code += "{0:#0{1}x}".format(rr_period, 4)[2:]
    gene_code += 12 * "0"
    return gene_code


def encode_movement_speed_gene(mov_speed, crosvr_mlt=0.5):
    code = "2"
    code += "{0:#0{1}x}".format(int(np.round(crosvr_mlt / max_crossover_mult * 15)), 1)[2:]  # crossover multiplier
    code += "{0:#0{1}x}".format(mov_speed, 4)[2:]
    code += 12 * "0"
    return code


def encode_connection_gene(src_node, dst_node, weight, bias, crosvr_mlt=0.5):
    if np.abs(weight) >= max_weight:
        weight = weight / np.abs(weight) * (max_weight - 0.001)
    if np.abs(bias) >= max_bias:
        bias = bias / np.abs(bias) * (max_bias - 0.001)
    gene_code = "0"
    gene_code += "{0:#0{1}x}".format(int(np.round(crosvr_mlt / max_crossover_mult * 15)), 1)[2:]  # crossover multiplier
    gene_code += "{0:#0{1}x}".format(src_node, 5)[2:]
    gene_code += "{0:#0{1}x}".format(dst_node, 5)[2:]
    gene_code += "{0:#0{1}x}".format(int(weight * 32768 / max_weight + 32768), 6)[2:]
    gene_code += "{0:#0{1}x}".format(int(bias * 32768 / max_bias + 32768), 6)[2:]
    return gene_code


def encode_random_path_connection_gene(src_node, dst_node, weight, no_intermediate, crosvr_mlt=0.5):
    if np.abs(weight) >= max_weight:
        weight = weight / np.abs(weight) * (max_weight - 0.001)
    if no_intermediate > rpath_max_im:
        no_intermediate = rpath_max_im
    gene_code = "4"
    gene_code += "{0:#0{1}x}".format(int(np.round(crosvr_mlt / max_crossover_mult * 15)), 1)[2:]  # crossover multiplier
    gene_code += "{0:#0{1}x}".format(src_node, 5)[2:]
    gene_code += "{0:#0{1}x}".format(dst_node, 5)[2:]
    gene_code += "{0:#0{1}x}".format(int(weight * 32768 / max_weight + 32768), 6)[2:]
    gene_code += "{0:#0{1}x}".format(int(no_intermediate), 6)[2:]
    return gene_code


def encode_influence_connection_gene(src_node, dst_node, weight, crosvr_mlt=0.5, backward=True):
    if np.abs(weight) >= max_weight:
        weight = weight / np.abs(weight) * (max_weight - 0.001)
    if backward:
        gene_code = "5"
    else:
        gene_code = "6"
    gene_code += "{0:#0{1}x}".format(int(np.round(crosvr_mlt / max_crossover_mult * 15)), 1)[2:]  # crossover multiplier
    gene_code += "{0:#0{1}x}".format(src_node, 5)[2:]
    gene_code += "{0:#0{1}x}".format(dst_node, 5)[2:]
    gene_code += "{0:#0{1}x}".format(int(weight * 32768 / max_weight + 32768), 6)[2:]
    gene_code += "{0:#0{1}x}".format(int(0), 6)[2:]
    return gene_code


def encode_reinforcement_connection_gene(src_node, dst_node, weight, bias, crosvr_mlt=0.5):
    """
    Connection gene which looks for the strongest paths between a src and dst node and adds the specified weight/bias
    to it. If no path exists, the reinforcement gene will make a direct connection from the src to dst
    :param src_node:
    :param dst_node:
    :param weight:
    :param bias:
    :param crosvr_mlt:
    :return:
    """
    if np.abs(weight) >= max_weight:
        weight = weight / np.abs(weight) * (max_weight - 0.001)
    if np.abs(bias) >= max_bias:
        bias = bias / np.abs(bias) * (max_bias - 0.001)
    gene_code = "7"
    gene_code += "{0:#0{1}x}".format(int(np.round(crosvr_mlt / max_crossover_mult * 15)), 1)[2:]  # crossover multiplier
    gene_code += "{0:#0{1}x}".format(src_node, 5)[2:]
    gene_code += "{0:#0{1}x}".format(dst_node, 5)[2:]
    gene_code += "{0:#0{1}x}".format(int(weight * 32768 / max_weight + 32768), 6)[2:]
    gene_code += "{0:#0{1}x}".format(int(bias * 32768 / max_bias + 32768), 6)[2:]
    return gene_code


def get_crossover_mult(gene_code, normalize=True):
    if normalize:
        return int(gene_code[1], 16) * max_crossover_mult / 15
    else:
        return int(gene_code[1], 16)


def decode_clk_init_gene(gene_code):
    crosvr_mlt = int(gene_code[1], 16) * max_crossover_mult / 15
    clk_limit = int(gene_code[2:4], 16)
    return clk_limit, crosvr_mlt


def decode_decision_threshold_gene(gene_code):
    crosvr_mlt = int(gene_code[1], 16) * max_crossover_mult / 15
    dec_thresh = int(gene_code[2:4], 16)/255.
    return dec_thresh, crosvr_mlt


def decode_rr_init_gene(gene_code):
    crosvr_mlt = int(gene_code[1], 16) * max_crossover_mult / 16
    rr_init = int(gene_code[2:4], 16)
    return rr_init, crosvr_mlt


def decode_movement_speed_gene(gene_code):
    crosvr_mlt = int(gene_code[1], 16) * max_crossover_mult / 15
    movement_speed = int(gene_code[2:4], 16)
    return movement_speed, crosvr_mlt


def decode_connection_gene(gene_code):
    crosvr_mlt = int(gene_code[1], 16) * max_crossover_mult / 15
    src_node = int(gene_code[2:5], 16)
    dst_node = int(gene_code[5:8], 16)
    weight = (int(gene_code[8:12], 16) - 32768.) / (32768. / max_weight)
    bias = (int(gene_code[12:16], 16) - 32768.) / (32768. / max_bias)
    return src_node, dst_node, weight, bias, crosvr_mlt


def decode_random_path_connection_gene(gene_code):
    crosvr_mlt = int(gene_code[1], 16) * max_crossover_mult / 15
    src_node = int(gene_code[2:5], 16)
    dst_node = int(gene_code[5:8], 16)
    weight = (int(gene_code[8:12], 16) - 32768.) / (32768. / max_weight)
    no_intermediate = int(gene_code[12:16], 16)
    return src_node, dst_node, weight, no_intermediate, crosvr_mlt


def decode_influence_connection_gene(gene_code):
    crosvr_mlt = int(gene_code[1], 16) * max_crossover_mult / 15
    src_node = int(gene_code[2:5], 16)
    dst_node = int(gene_code[5:8], 16)
    weight = (int(gene_code[8:12], 16) - 32768.) / (32768. / max_weight)
    if gene_code[0] == "5":
        backward = True
    elif gene_code[0] == "6":
        backward = False
    else:
        print("WARNING: Invalid influence gene code: {}".format(gene_code[0]))
        return None
    return src_node, dst_node, weight, crosvr_mlt, backward


def decode_reinforcment_connection_Gene(gene_code):
    crosvr_mlt = int(gene_code[1], 16) * max_crossover_mult / 15
    src_node = int(gene_code[2:5], 16)
    dst_node = int(gene_code[5:8], 16)
    weight = (int(gene_code[8:12], 16) - 32768.) / (32768. / max_weight)
    bias = (int(gene_code[12:16], 16) - 32768.) / (32768. / max_bias)
    return src_node, dst_node, weight, bias, crosvr_mlt


def summarize_connection_gene(gene_code, input_names=None, output_names=None, node_counts=None, print_summary=True):
    src_node, dst_node, weight, bias, crosvr_mlt = decode_connection_gene(gene_code)

    if input_names is not None and output_names is not None and node_counts is not None:
        if dst_node < (len(input_names) + node_counts['conduit']):
            output_node_name = "C{}".format(dst_node - len(input_names))
        elif dst_node < (len(input_names) + node_counts['conduit'] + node_counts['hidden']):
            output_node_name = "H{}".format(dst_node - len(input_names)-node_counts['conduit'])
        else:
            output_node_name = output_names[dst_node - len(input_names) - node_counts['conduit'] - node_counts['hidden']]

        if src_node < len(input_names):
            input_node_name = input_names[src_node]
        elif src_node < len(input_names) + node_counts['conduit']:
            input_node_name = "C{}".format(src_node - len(input_names))
        else:
            input_node_name = "H{}".format(src_node - len(input_names) - node_counts['conduit'])

        summary = "src_node = {}, dst_node = {}, weight = {}, bias = {}, crossover = {}".format(input_node_name,
                                                                                                output_node_name,
                                                                                                weight, bias,
                                                                                                crosvr_mlt)
    else:
        summary = "src_node = {}, dst_node = {}, weight = {}, bias = {}, crossover = {}".format(src_node, dst_node,
                                                                                                weight, bias,
                                                                                                crosvr_mlt)
    if print_summary:
        print(summary)
    else:
        return summary


def summarize_gene(gene_code, input_names=None, output_names=None, node_counts=None,
                   template=None, print_summary=True):

    if template is not None:
        output_names = template.output_names
        if template.data_input_count is not None:
            input_names = []
            for name in template.input_names:
                input_names.append(name)
            for i in range(template.data_input_count):
                input_names.append("data {}".format(i))
            if template.position_count is not None:  # Position density input names
                for i in range(template.position_count):
                    input_names.append("Position Density {} - 1".format(i))
                for i in range(template.position_count):
                    input_names.append("Position Density {} - 0".format(i))
        else:
            input_names = template.input_names

    if gene_code[0] == '0':  # connection gene
        summary = "{} (connection): ".format(gene_code) + summarize_connection_gene(gene_code, input_names,
                                                                                    output_names,
                                                                                    node_counts=node_counts,
                                                                                    print_summary=False)
    elif gene_code[0] == '1':  # internal timer initial period gene
        clk_init, crossover = decode_clk_init_gene(gene_code)
        summary = "{} (clk init limit): ".format(gene_code) + "period = {}, crossover = {}".format(clk_init, crossover)
    elif gene_code[0] == '2':  # initial movement speed gene
        mov_speed, crossover = decode_movement_speed_gene(gene_code)
        summary = "{} (mov speed init): ".format(gene_code) + "speed = {}, crossover = {}".format(mov_speed, crossover)
    elif gene_code[0] == '3':  # initial reaction rate gene
        rr_init, crossover = decode_rr_init_gene(gene_code)
        summary = "{} (reaction rate init): ".format(gene_code) + "period = {}, crossover = {}".format(rr_init, crossover)
    elif gene_code[0] == '4':
        src_node, dst_node, weight, no_intermediate, crosvr_mlt = decode_random_path_connection_gene(gene_code)
        summary = "{} (random connection path): ".format(gene_code) + \
                  "src node = {}, dst node = {}, # intermediate = {}, weight = {}".format(src_node, dst_node, no_intermediate, weight)
    elif gene_code[0] == '5':
        src_node, dst_node, weight, crosvr_mlt, _ = decode_influence_connection_gene(gene_code)
        summary = "{} (backward connection influence): ".format(gene_code) + \
                  "src node = {}, dst node = {}, weight = {}".format(src_node, dst_node, weight)
    elif gene_code[0] == '6':
        src_node, dst_node, weight, crosvr_mlt, _ = decode_influence_connection_gene(gene_code)
        summary = "{} (forward connection influence): ".format(gene_code) + \
                  "src node = {}, dst node = {}, weight = {}".format(src_node, dst_node, weight)
    elif gene_code[0] == '7':
        src_node, dst_node, weight, bias, crosvr_mlt = decode_reinforcment_connection_Gene(gene_code)
        summary = "{} (reinforcement connection): ".format(gene_code) + \
                  "src node = {}, dst node = {}, weight = {}, bias = {}".format(src_node, dst_node, weight, bias)
    elif gene_code[0] == '8':
        dec_thresh, crosovr_mlt = decode_decision_threshold_gene(gene_code)
        summary = "{} (decision thresh init): ".format(gene_code) + "thresh = {}, crossover = {}".format(dec_thresh, crosovr_mlt)
    elif gene_code[0] == "f":
        summary = "{} (latent gene): ".format(gene_code)
    else:
        summary = "{} : INVALID GENE TYPE".format(gene_code)
    if print_summary:
        print(summary)
    else:
        return summary


def summarize_genome(genes=None, organism=None, input_names=None, output_names=None, node_counts=None,
                     species_template=None):
    summary = ""
    if species_template is not None:
        input_names = species_template.input_names
        output_names = species_template.output_names
    if organism is not None:
        genes = organism.genes
        internal_node_count = organism.brain.internal_neuron_count
    for g in genes:
        summary += summarize_gene(gene_code=g, input_names=input_names, output_names=output_names,
                                  node_counts=node_counts, print_summary=False, template=species_template) + "\n"
    return summary


def generate_gene_pool(organisms, mutation_rate, ignore_alive=False):
    genes = []
    for org in organisms:
        if org.alive or ignore_alive:
            genes += org.genes
    for i in range(len(genes)):
        if np.random.rand() < mutation_rate:
            genes[i] = random_gene(node_counts=organisms[0].brain.node_counts)
    return genes


def apply_mutation(gene, node_counts, max_CLK_lim, max_RR_lim=None):
    """
    Apply a random mutation to a gene, with a small chance to generate an entirely new gene.
    :param gene: [hex string] gene code before mutation
    :param node_counts:
    :param max_CLK_lim:
    :param max_RR_lim:
    :return: [hex string] mutated gene code
    """

    if np.random.rand() < 0.05:  # generate completely new gene
        gene = random_gene(node_counts=node_counts)
    else:
        if gene[0] == '0' or gene[0] == '7':  # connection gene or reinforcement connection
            if gene[0] == '0':
                src_node, dst_node, weight, bias, crosvr_mlt = decode_connection_gene(gene)
            elif gene[0] == '7':
                src_node, dst_node, weight, bias, crosvr_mlt = decode_random_path_connection_gene(gene)
            else:
                print("ERROR - Apply_Mutation(): AJA7")
                return None
            r_param = np.random.rand()
            if r_param > 0.8:  # mutate src node
                src_node = get_random_node(node_counts=node_counts, node_code="ichm")
                if check_node_type(src_node, node_counts, "c"):  # src_node is conduit neuron
                    # Check if dst node is valid
                    if check_node_type(dst_node, node_counts, "ic"):  # invalid dst node (conduit destination)
                        dst_node = get_random_node(node_counts=node_counts, node_code="hmo")
            elif r_param > 0.6:  # mutate dst node
                if check_node_type(src_node, node_counts, "c"):  # src_node is conduit neuron
                    dst_node = get_random_node(node_counts=node_counts, node_code="hmo")
                else:  # src node is not conduit node
                    if np.random.rand() > 0.5:  # destination: output node
                        dst_node = get_random_node(node_counts=node_counts, node_code="o")
                    else:  # destination: internal node
                        dst_node = get_random_node(node_counts=node_counts, node_code="chm")
            elif r_param > 0.4:
                weight = np.random.rand() * 2 * max_weight - max_weight
            elif r_param > 0.2:
                bias = np.random.rand() * 2 * max_bias - max_bias
            else:
                crosvr_mlt = np.random.rand()
            if gene[0] == '0':
                gene = encode_connection_gene(src_node, dst_node, weight, bias, crosvr_mlt)
            elif gene[0] == '7':
                gene = encode_reinforcement_connection_gene(src_node, dst_node, weight, bias, crosvr_mlt)
            else:
                print("ERROR - Apply_Mutation(): AJA7")
        elif gene[0] == '1':  # internal timer initial period gene
            timer_period, crosvr_mlt = decode_clk_init_gene(gene)
            if np.random.rand() >= 0.5:  # mutate period
                timer_period = int(np.round(np.random.rand() * (max_CLK_lim - base_min_CLK_lim) + base_min_CLK_lim))
            else:  # mutate crossover multiplier
                crosvr_mlt = np.random.rand()
            gene = encode_clk_init_gene(timer_period, crosvr_mlt)
        elif gene[0] == '2':  # initial movement speed gene
            mov_speed, crosvr_mlt = decode_movement_speed_gene(gene)
            if np.random.rand() >= 0.5:  # mutate move_speed
                mov_speed = int(np.round(np.random.rand() * (max_mvmt_speed - min_mvmt_speed) + min_mvmt_speed))
            else:  # mutate crossover multiplier
                crosvr_mlt = np.random.rand()
            gene = encode_movement_speed_gene(mov_speed, crosvr_mlt)
        elif gene[0] == '3':  # internal timer initial period gene
            rr_period, crosvr_mlt = decode_rr_init_gene(gene)
            if np.random.rand() >= 0.5:  # mutate period
                rr_period = int(np.round(np.random.rand() * (max_RR_lim - base_min_RR_lim) + base_min_RR_lim))
            else:  # mutate crossover multiplier
                crosvr_mlt = np.random.rand()
            gene = encode_clk_init_gene(rr_period, crosvr_mlt)
        elif gene[0] == '4':  # random connection path gene
            src_node, dst_node, weight, no_intermediate, crosvr_mlt = decode_random_path_connection_gene(gene)
            r_param = np.random.rand()
            if r_param > 0.8:  # mutate src node
                src_node = get_random_node(node_counts=node_counts, node_code="ihcm")
                if check_node_type(src_node, node_counts, "c"):  # src_node is conduit neuron
                    # Check if dst node is valid
                    if check_node_type(dst_node, node_counts, "ic"):  # invalid dst node (conduit destination)
                        dst_node = get_random_node(node_counts=node_counts, node_code="hmo")
            elif r_param > 0.6:  # mutate dst node
                if check_node_type(src_node, node_counts, "c"):  # src_node is conduit neuron
                    dst_node = get_random_node(node_counts=node_counts, node_code="hmo")
                else:  # src node is not conduit node
                    if np.random.rand() > 0.5:  # destination: output node
                        dst_node = get_random_node(node_counts=node_counts, node_code="o")
                    else:  # destination: internal node
                        dst_node = get_random_node(node_counts=node_counts, node_code="chm")
            elif r_param > 0.4:  # mutate weight
                weight = np.random.rand() * 2 * max_weight - max_weight
            elif r_param > 0.2:
                no_intermediate = np.random.randint(1, rpath_max_im)
            else:
                crosvr_mlt = np.random.rand()
            gene = encode_random_path_connection_gene(src_node=src_node, dst_node=dst_node, weight=weight,
                                                      no_intermediate=no_intermediate, crosvr_mlt=crosvr_mlt)
        elif gene[0] == '5' or gene[0] == '6':  # random connection influence gene
            src_node, dst_node, weight, crosvr_mlt, backward = decode_influence_connection_gene(gene)
            r_param = np.random.rand()
            if r_param > 0.7:  # mutate src node
                src_node = get_random_node(node_counts=node_counts, node_code="ichm")
                if check_node_type(src_node, node_counts, "c"):  # src_node is conduit neuron
                    # Check if dst node is valid
                    if check_node_type(dst_node, node_counts, "ic"):  # invalid dst node (conduit destination)
                        dst_node = get_random_node(node_counts=node_counts, node_code="hmo")
            elif r_param > 0.4:  # mutate dst node
                if check_node_type(src_node, node_counts, "c"):  # src_node is conduit neuron
                    dst_node = get_random_node(node_counts=node_counts, node_code="hmo")
                else:  # src node is not conduit node
                    if np.random.rand() > 0.5:  # destination: output node
                        dst_node = get_random_node(node_counts=node_counts, node_code="o")
                    else:  # destination: internal node
                        dst_node = get_random_node(node_counts=node_counts, node_code="chm")
            elif r_param > 0.2:  # mutate weight
                weight = np.random.rand() * 2 * max_weight - max_weight
            else:
                crosvr_mlt = np.random.rand()
            gene = encode_influence_connection_gene(src_node=src_node, dst_node=dst_node, weight=weight,
                                                    backward=backward, crosvr_mlt=crosvr_mlt)
        elif gene[0] == '8':  # initial decision threshold gene
            dec_thresh, crosvr_mlt = decode_decision_threshold_gene(gene)
            if np.random.rand() >= 0.5:  # mutate threshold
                dec_thresh = np.random.rand() * (base_max_dec_thresh - base_min_dec_thresh) + base_min_dec_thresh
            else:  # mutate crossover multiplier
                crosvr_mlt = np.random.rand()
            gene = encode_decision_threshold_init_gene(dec_thresh, crosvr_mlt)
        elif gene[0] == 'f':
            return gene
        else:
            print("ERROR (Apply_Mutation): Invalid gene code ({})".format(gene[0]))
    return gene


def gene_distance(gene1, gene2):
    dist = 0
    for i in range(len(gene1)):
        dist += np.abs(int("0x" + gene1[i], 0) - int("0x" + gene2[i], 0))
    return dist / (len(gene1) * 15)


def shared_genes(genome1, genome2, no_genes):
    genes1 = list(chunk(genome1, no_genes))
    genes2 = list(chunk(genome2, no_genes))

    matches = 0
    for g1 in genes1:
        if g1 in genes2:
            matches += 1
    return matches / no_genes


def random_mix_genes(genes1, genes2, crossover_rate=0.5):
    genes_new = []
    for i in range(len(genes1)):
        if np.random.rand() >= crossover_rate:
            genes_new.append(genes1[i])
        else:
            genes_new.append(genes2[i])
    return genes_new

'''
# multiplier-based crossover
def Random_Splice_Genes(genomes, rnd_array, rnd_idx, crossover_rate=0.5, skip_rate=None, dupl_rate=None):
    genome_new = []
    src_parent_idx = 0
    if rnd_array[rnd_idx] >= crossover_rate:
        src_parent_idx = 1
    rnd_idx += 1
    for i in range(len(genomes[0])):
        genome_new.append(genomes[src_parent_idx][i])
        crosvr_mlt = Get_Crossover_Mult(genomes[src_parent_idx][i], normalize=True)
        #crosvr_mlt = 1
        if rnd_array[rnd_idx] <= crossover_rate * crosvr_mlt:
            src_parent_idx += 1
            if src_parent_idx >= len(genomes):
                src_parent_idx = 0
        rnd_idx += 1
    return genome_new
'''


# multiplier-based crossover using duplication + skip
def random_splice_genes(genomes, rnd_array, rnd_idx, crossover_rate=0.5, skip_rate=0.00, dupl_rate=0.00):
    genome_new = []
    src_parent_idx = 0
    if rnd_array[rnd_idx] >= crossover_rate:
        src_parent_idx = 1
    rnd_idx += 1
    gene_ptr = 0
    for i in range(len(genomes[0])):
        genome_new.append(genomes[src_parent_idx][gene_ptr])
        crosvr_mlt = get_crossover_mult(genomes[src_parent_idx][gene_ptr], normalize=True)
        #crosvr_mlt = 1
        if rnd_array[rnd_idx] <= crossover_rate * crosvr_mlt:
            src_parent_idx += 1
            if src_parent_idx >= len(genomes):
                src_parent_idx = 0
        rnd_idx += 1
        gene_ptr += 1
        if dupl_rate > 0 or skip_rate > 0:
            if np.random.rand() < dupl_rate:  # duplicate previous gene
                gene_ptr -= 1
            elif np.random.rand() < skip_rate:  # skip next gene
                gene_ptr += 1
        if gene_ptr >= len(genomes[0]):  # if at end of genome, return pointer to 0
            gene_ptr = 0
    return genome_new


'''
# key-based crossover
def Random_Splice_Genes(genomes, rnd_array, rnd_idx, crossover_rate=0.5, dupl_rate=0.00, skip_rate=0.00):
    genome_new = []
    src_parent_idx = 0
    if rnd_array[rnd_idx] >= crossover_rate:
        src_parent_idx = 1
    rnd_idx += 1
    gene_ptr = 0
    for i in range(len(genomes[0])):
        genome_new.append(genomes[src_parent_idx][gene_ptr])
        crosvr_key = Get_Crossover_Mult(genomes[src_parent_idx][gene_ptr], normalize=False)
        crosv_key_start = (crosvr_key >> 0) & ((1 << 2)-1)  # gene start key
        crosv_key_end = (crosvr_key >> 2) & ((1 << 2)-1)  # gene end key

        if i < (len(genomes[0])-1) and gene_ptr < (len(genomes[0])-1):
            crosvr_key_next = Get_Crossover_Mult(genomes[src_parent_idx][gene_ptr+1], normalize=False)
            crosvr_key_start_next = (crosvr_key_next >> 0) & ((1 << 2)-1)  # next gene start key
        else:
            crosvr_key_start_next = -1

        # if gene start/end keys are equal or if the gene end key doesn't match the next gene start key, crossover
        if crosv_key_start == crosv_key_end or not crosv_key_end == crosvr_key_start_next:
            if rnd_array[rnd_idx] <= crossover_rate:
                src_parent_idx += 1
                if src_parent_idx >= len(genomes):
                    src_parent_idx = 0
        rnd_idx += 1
        gene_ptr += 1
        if np.random.rand() < dupl_rate:  # duplicate previous gene
            gene_ptr -= 1
        elif np.random.rand() < skip_rate:  # skip next gene
            gene_ptr += 1
        if gene_ptr >= len(genomes[0]):  # if at end of genome, return pointer to 0
            gene_ptr = 0
    return genome_new
'''


def check_genome_has_connections(organism, in_out_idx_list, genome=None):
    """
    For a list of input-output pairs of connection node indexes, check if the genome contains any connections
    :param organism: organism with genome (list of genes)
    :param genome: genome (list of genes)
    :param in_out_idx_list: list of connection indexes to check for [[in_idx1, out_idx1, wght1], [in_idx2, out_idx2, wght1], ...]
    where wght is +1 or -1
    :return:
    """
    if organism is not None:
        genome = organism.genes
    found_connection = np.zeros(len(in_out_idx_list))
    weights = np.zeros(len(in_out_idx_list))
    for i in range(len(in_out_idx_list)):
        for g in genome:
            if g[0] == '0':  # check if is connection gene
                src_node, dst_node, weight, bias, crosvr_mlt = decode_connection_gene(g)
                if src_node == in_out_idx_list[i][0] and dst_node == in_out_idx_list[i][1] and np.sign(weight) == np.sign(in_out_idx_list[i][2]):
                    found_connection[i] = 1
                    weights[i] = weight
    return found_connection, weights


def genetic_histogram(gene_pool, max_show_genes=150, save_path=None, save_text_file=None, no_internal_nodes=None, template=None):
    """
    Generate a histogram distribution of a gene pool showing gene occurrences. Option to save or plot histogram
    :param gene_pool:
    :param save_path:
    :return:
    """
    from collections import Counter

    counts = Counter(gene_pool)
    common = counts.most_common()
    labels = [item[0] for item in common]
    number = [item[1] for item in common]

    if save_path is not None:
        nbars = len(common)
        plt.figure(figsize=(20, 20))
        if nbars > max_show_genes:
            plt.bar(np.arange(max_show_genes), number[:max_show_genes], tick_label=labels[:max_show_genes])
        else:
            plt.bar(np.arange(nbars), number, tick_label=labels)
        plt.xticks(rotation=90)
        plt.yscale('log')
        if save_path == "show":
            plt.show()
        else:
            plt.savefig(save_path)
    plt.close()

    if save_text_file is not None:
        with open(save_text_file, 'w') as f:
            for i in range(10):
                f.write('{} ({}): {}\n'.format(i, number[i], summarize_gene(gene_code=labels[i],
                                                                            template=template,
                                                                            internal_node_count=no_internal_nodes,
                                                                            print_summary=False)))

    return number, labels


def get_random_node(node_counts, node_code, node_order=("input", "conduit", "hidden", "memory", "output")):
    #total_nodes = sum(node_counts.values())

    if not node_code:
        node_code = ''.join(node_counts.keys())

    node_ranges = {}
    running_count = 0
    for key in node_order:
        node_ranges[key] = [i + running_count for i in range(node_counts[key])]
        running_count += node_counts[key]

    valid_indices = []
    for i, (_, count) in enumerate(list(node_counts.items())[:-1]):
        if node_order[i][0] in node_code:
            valid_indices.extend(node_ranges[node_order[i]])

    if not valid_indices:
        raise ValueError("Invalid code provided")

    return random.choice(valid_indices)


def check_node_type(node_idx, node_counts, node_code, node_order=("input", "conduit", "hidden", "memory", "output")):
    #total_nodes = sum(node_counts.values())

    if not node_code:
        node_code = ''.join(node_counts.keys())

    node_ranges = {}
    running_count = 0
    for key in node_order:
        node_ranges[key] = [i + running_count for i in range(node_counts[key])]
        running_count += node_counts[key]

    for i, (_, node_range) in enumerate(node_ranges.items()):
        if node_order[i][0] in node_code and node_idx in node_range:
            return True

    return False



def genetic_diversity(gene_pool):
    from collections import Counter

    counts = Counter(gene_pool)
    common = counts.most_common()

    return len(common)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    node_counts = {"input": 16, "output":8, "hidden":6, "conduit":5, "memory": 2}
    node_counts["total"] = np.array([node_counts[k] for k in node_counts.keys()]).sum()

    rn = get_random_node(node_counts, node_code="o")
    check_node_type(node_idx=0, node_counts=node_counts, node_code="o")

    # Test reaction period gene
    rr_gene = generate_random_rr_init_gene()
    summarize_gene(rr_gene)

    # Test gene histogram
    import pickle

    with open("../simState_simple2D_corners.p", 'rb') as handle:
        data = pickle.load(handle)
    gene_pool = data[0]
    print("genes count: {}".format(len(gene_pool)))

    genetic_histogram(gene_pool, "./hist.png")

    # Test calculating gene distance
    code1 = generate_random_connection_gene(node_counts=node_counts)
    code2 = generate_random_connection_gene(node_counts=node_counts)
    code1 = '0207f6a2337a00'
    code2 = '1197f6a2337a00'
    d = gene_distance(code1, code2)
    print("{} : {} => {}".format(code1, code2, d))

    # Test calculating number of shared genes
    gene1 = '0207f6a2337a00'
    gene2 = '1197f6a2337a00'
    gene3 = '1197f6a2337a00'
    gene4 = '1397f552337a00'
    gene5 = '1197f6a436fa00'
    gene6 = '1111f442337a00'
    genome1 = gene1 + gene2 + gene6 + gene5
    genome2 = gene2 + gene3 + gene4 + gene6
    print("genomes")
    print(genome1)
    print(genome2)
    print("matches: {}".format(shared_genes(genome1, genome2, 4)))

    print("Random CLK period timer code: " + generate_random_clk_init_gene())
