import numpy as np
import time
import datetime
from src.graphs import generate_graph
from src.genome import generate_random_connection_gene, encode_connection_gene, decode_connection_gene, \
    summarize_connection_gene, decode_random_path_connection_gene, decode_influence_connection_gene, \
    decode_reinforcment_connection_Gene


# inputs:
# oscillator neuron: input with oscillating value
# genetic similarity sensory input
# previous movement
# age
# pheromones: magnitude, gradient in different directions

# outputs:
# emit pheromone
# change oscillation period


def hex_to_binary(hex_number: str, num_digits: int = 8) -> str:
    """
    Converts a hexadecimal value into a string representation
    of the corresponding binary value
    Args:
        hex_number: str hexadecimal value
        num_digits: integer value for length of binary value.
                    defaults to 8
    Returns:
        string representation of a binary number 0-padded
        to a minimum length of <num_digits>
    """
    return str(bin(int(hex_number, 16)))[2:].zfill(num_digits)


def find_connected(connections, node_count, src_node=None, dst_node=None, max_connected=100):
    """
    Find all nodes connected either forward to a specified destination (dst) node or backwards to a specified (src) node
    or both. Return a vector of shape [node_count] where '0' means that node is not connected and '1' means that it is.
    :param connections: List of connection objects
    :param node_count: (int) count of nodes
    :param src_node: (int) source node index
    :param dst_node: (int) destination node index
    :return: Numpy array of shape [node_count] population with '0's (node not connected) and '1's (node connected)
    """
    connected_to_dst = np.ones(node_count).astype('int')
    connected_to_src = np.ones(node_count).astype('int')
    if dst_node is not None:
        connected_to_dst = np.zeros(node_count).astype('int')
        #connected_to_dst[dst_node] = 1
        # Check for forward connections: connections leading to a specified destination
        remaining_connections = True
        while remaining_connections:  # If the last iteration added a new connected node, keep looking
            remaining_connections = False  # If there are no more connected nodes, this will stay 'False'
            for conn in connections:
                # For each connection, check if the destination/target of the connection is either the specified
                # destination or is connected to the destination.
                if conn.dst_node == dst_node or connected_to_dst[conn.dst_node] == 1:
                    # Set the source node (we are working backward through the graph) to '1'
                    if connected_to_dst[conn.src_node] == 0:
                        connected_to_dst[conn.src_node] = 1
                        remaining_connections = True

    if src_node is not None:
        connected_to_src = np.zeros(node_count).astype('int')
        #connected_to_src[src_node] = 1
        # Check for backward connections: connections leading to a specified source node
        remaining_connections = True
        while remaining_connections:  # If the last iteration added a new connected node, keep looking
            remaining_connections = False  # If there are no more connected nodes, this will stay 'False'
            for conn in connections:
                # For each connection, check if the source of the connection is either the specified
                # source node or is connected to the source node.
                if conn.src_node == src_node or connected_to_src[conn.src_node] == 1:
                    # Set the destination node (we are working forwards through the graph) to '1'
                    if connected_to_src[conn.dst_node] == 0:
                        connected_to_src[conn.dst_node] = 1
                        remaining_connections = True
    # If both target source and destination nodes are specified, multiple to get only the nodes connected to both.
    connected = connected_to_dst * connected_to_src
    return connected


def find_paths(connections, src, dst):
    """
    Given the list of connections, find all the possible paths from the src node to the dst node, ignoring looped paths.
    Return a list of paths, the weights of the connections of each path, and the connection indexes of each path.
    :param connections:
    :param src:
    :param dst:
    :return:
    """
    def dfs(node, visited, path, path_weights, path_conn_idx, paths, paths_weights, paths_conn_idx, max_paths=100):
        # mark the current node as visited
        visited.add(node)
        path.append(node)

        # if the current node is the destination node, add the path to the paths list
        if node == dst:
            paths.append(list(path))
            paths_weights.append(list(path_weights))
            paths_conn_idx.append(list(path_conn_idx))
        else:
            # for each connection from the current node
            for cidx, conn in enumerate(connections):
                if conn.src_node == node:
                    # if the destination node is not visited, recursively call dfs
                    if conn.dst_node not in visited:
                        path_weights.append(conn.weight)
                        path_conn_idx.append(cidx)
                        if len(paths) < max_paths:
                            dfs(conn.dst_node, visited, path, path_weights, path_conn_idx, paths, paths_weights, paths_conn_idx)
                        path_weights.pop()
                        path_conn_idx.pop()

        # backtrack by removing the current node from the path
        path.pop()
        #path_weights.pop()
        visited.remove(node)

    paths = []
    paths_weights = []
    paths_conn_idx = []
    visited = set()
    dfs(src, visited, [], [], [], paths, paths_weights, paths_conn_idx)
    #if len(paths) > 0:
    #    print(len(paths))
    return paths, paths_weights, paths_conn_idx


def random_path_influence(connections, src_node, dst_node, no_hidden_stops, node_counts, weight=0):
    """
    For a path connection between the src and dst node with randomly selected intermediate hidden layers.
    :param connections:
    :param src_node:
    :param dst_node:
    :param max_no_stops:
    :param min_no_stops:
    :param input_count:
    :param self.node_counts['output']:
    :param internal_count:
    :param conduit_count:
    :return:
    """
    if (node_counts['hidden']+node_counts['conduit']+node_counts['memory']) > 0:
        internal_nodes = []
        for i in range(no_hidden_stops):
            internal_nodes.append(np.random.randint(node_counts['input'],
                                                    node_counts['input']+node_counts['hidden']+node_counts['conduit']+node_counts['memory']))

        connections.append(Connection(src_node=src_node, dst_node=internal_nodes[0], weight=weight))
        for i in range(1, no_hidden_stops):
            connections.append(Connection(src_node=internal_nodes[i-1], dst_node=internal_nodes[i], weight=weight))
        connections.append(Connection(src_node=internal_nodes[-1], dst_node=dst_node, weight=weight))
    else:
        connections.append(Connection(src_node=src_node, dst_node=dst_node, weight=weight))



def apply_influence(dst_node, connections, node_counts, src_node=None,
                    weight=0, backward="True"):
    connected_nodes_dst = find_connected(connections=connections, node_count=node_counts['total'], src_node=None,
                                         dst_node=dst_node)
    connected_nodes_src = find_connected(connections=connections, node_count=node_counts['total'], src_node=src_node,
                                         dst_node=None)

    if src_node is None:
        # Amplify all connections to nodes already connected to the specified destinations
        # Does not create new connections
        # NOT USED
        for n in range(node_counts['total']):
            pass
    else:
        if not backward:
            # Forward influence method
            # Amplify existing connections
            for n in range(node_counts['total']):
                if connected_nodes_dst[n] == 1:  # If node is connected to influence destination
                    if connected_nodes_src[n] == 1:  # If it's already connected to the source, amplify both connections
                        if n >= node_counts['total']:
                            connections.append(Connection(src_node=src_node, dst_node=n, weight=weight))
                        connections.append(Connection(src_node=n, dst_node=dst_node, weight=weight))
                    elif not n == src_node and n >= node_counts['total']:  # If it's not yet connected to the src node, add a new connection there only
                        connections.append(Connection(src_node=src_node, dst_node=n, weight=weight))
        else:
            # Backward influence method
            for n in range(node_counts['total']):
                if connected_nodes_src[n] == 1:  # If node is connected to influence source
                    if connected_nodes_dst[n] == 1:  # if it's already connected to the destination node, amplify both
                        if n >= node_counts['input']:
                            connections.append(Connection(src_node=src_node, dst_node=n, weight=weight))
                        connections.append(Connection(src_node=n, dst_node=dst_node, weight=weight))
                    else:
                        connections.append(Connection(src_node=n, dst_node=dst_node, weight=weight))


def apply_reinforcement(src, dst, weight, bias, connections):
    """
    Apply reinforcement gene to existing connections
    First, identify all existing paths from src to dst.
    Amplify/attenuate the strongest existing path. If no path exists, make a new direct path from src to dst.
    :param src:
    :param dst:
    :param weight:
    :param bias:
    :param connections:
    :return:
    """
    # Get all paths from src to dst
    paths, weights, conn_idx = find_paths(connections=connections, src=src, dst=dst)

    # if no existing path, add new connection from src to dst
    if len(paths) == 0:
        connections.append(Connection(src_node=src, dst_node=dst, weight=weight, bias=bias))
    elif len(paths) == 1 and len(paths[0]) == 1:
        connections.append(Connection(src_node=src, dst_node=dst, weight=weight, bias=bias))
    else:
        path_weights = [np.array(w).mean() for w in weights]
        strongest_path = np.argmax(path_weights)
        for cidx in conn_idx[strongest_path]:
            connections[cidx].weight += weight
            connections[cidx].bias += bias


class Brain:
    def __init__(self, node_counts, max_connections):
        super(Brain, self).__init__()
        self.max_connections = max_connections
        self.node_counts = node_counts
        self.node_counts['total'] = self.node_counts['total']
        self.connections = []

    def forward(self, inputs):

        # signal structure: [inputs, conduit, hidden, memory, outputs]
        node_outputs_prv = np.zeros((self.node_counts['total']))
        # node_outputs = np.zeros((self.node_counts['total']))
        t0 = datetime.datetime.now()

        # assign inputs
        for i in range(len(inputs)):
            node_outputs_prv[i] = inputs[i]

        # assign hidden states
        for i in range(self.node_counts['hidden']):
            node_outputs_prv[len(inputs) + self.node_counts['conduit'] + i] = self.internal_state[i]

        # calculate and accumulate conduit neurons
        node_inputs = np.zeros((self.node_counts['total']))
        node_inputs_conduit_temp = np.zeros((self.node_counts['total']))
        for conn in self.connections:
            if conn.active:
                node_inputs_conduit_temp[conn.dst_node] += conn.calculate(node_outputs_prv[conn.src_node])
        node_inputs_conduit_temp = np.tanh(np.array(node_inputs_conduit_temp))
        # Copy conduit neuron values to previous output nodes
        node_outputs_prv[len(inputs):(len(inputs) + self.node_counts['conduit'])] = node_inputs_conduit_temp[len(inputs):(
                    len(inputs) + self.node_counts['conduit'])]

        # calculate and accumulate node inputs
        for conn in self.connections:
            if conn.active:
                node_inputs[conn.dst_node] += conn.calculate(node_outputs_prv[conn.src_node])

        # flatten inputs to calculate output
        node_outputs = np.tanh(np.array(node_inputs))

        # update internal state
        for i in range(self.node_counts['hidden']):
            self.internal_state[i] = node_outputs[len(inputs) + self.node_counts['conduit'] + i]

        # set and return organism outputs
        organism_outputs = np.zeros(self.node_counts['output'])
        for i in range(self.node_counts['output']):
            organism_outputs[i] = node_outputs[
                self.node_counts["input"] + self.node_counts['conduit'] + self.node_counts['hidden'] + i]
        return organism_outputs

    def get_node_state(self, inputs):
        node_state = np.zeros((self.node_counts['total']))
        # assign inputs
        for i in range(len(inputs)):
            node_state[i] = inputs[i]

        # assign hidden states
        for i in range(self.node_counts['hidden']):
            node_state[len(inputs) + self.node_counts['conduit'] + i] = self.internal_state[i]
        return node_state

    def Prune_Connections(self):
        """
        Determine if any hidden nodes don't contain routes to outputs, and if so, ignore any connections to these
        nodes
        :return: None
        """

        hidden_node_used = np.zeros(self.node_counts['hidden'])
        for conn in self.connections:
            if self.is_hidden_node(conn.src_node) and self.is_output_node(conn.dst_node):  # src=hidden, dst=output
                hidden_node_used[int(conn.src_node - self.node_counts['input'])] = 1
            elif self.is_hidden_node(conn.src_node) and self.is_hidden_node(conn.dst_node):  # src=hidden, dst=hidden
                hidden_node_used[int(conn.src_node - self.node_counts['input'])] = 1

        for conn in self.connections:
            if self.is_hidden_node(conn.src_node):
                if hidden_node_used[int(conn.src_node - self.node_counts['input'])] == 0:
                    conn.active = False
            if self.is_hidden_node(conn.dst_node):
                if hidden_node_used[int(conn.dst_node - self.node_counts['input'])] == 0:
                    conn.active = False

    def get_connections_mat(self):
        w = np.zeros((self.node_counts['total'], self.node_counts['total']))
        b = np.zeros((self.node_counts['total']))
        for conn in self.connections:
            w[conn.src_node, conn.dst_node] += conn.weight
            b[conn.dst_node] += conn.bias
        return w, b

    def random_init_connections(self):
        for n in range(self.max_connections):
            code = generate_random_connection_gene(node_counts=self.node_counts)
            self.add_connection(code)

    def add_connection(self, code):
        if code[0] == "0":  # single connection gene
            if len(self.connections) < self.max_connections:
                new_conn = Connection(code=code)
                if new_conn not in self.connections:
                    self.connections.append(new_conn)
                    # print("added new connection ({})".format(len(self.connections)))
                else:
                    print("ignored duplicate connection")
            else:
                print(
                    "Error (Cortex.py:Brain():add_connection()): tried to add new connection when there is already the maximum ({}) number of connections".format(
                        len(self.connections)))
        elif code[0] == "4":  # random path connection gene
            src_node, dst_node, weight, no_intermediate, crosvr_mlt = decode_random_path_connection_gene(code)
            random_path_influence(connections=self.connections, src_node=src_node, dst_node=dst_node,
                                  no_hidden_stops=no_intermediate, node_counts=self.node_counts, weight=weight)

        elif code[0] == "5" or code[0] == "6":  # influence connection gene

            src_node, dst_node, weight, crosvr_mlt, backward = decode_influence_connection_gene(code)
            apply_influence(node_counts=self.node_counts, src_node=src_node, dst_node=dst_node,
                            connections=self.connections, weight=weight, backward=backward)

        elif code[0] == "7":  # reinforcement connection gene
            src_node, dst_node, weight, bias, crosvr_mlt = decode_reinforcment_connection_Gene(code)
            apply_reinforcement(src=src_node, dst=dst_node, weight=weight, bias=bias, connections=self.connections)
        else:
            print("WARNING - Brain.add_connection(): Invalid connection gene code: {}".format(code[0]))

    def get_connection_genes(self):
        genes = []
        for conn in self.connections:
            genes.append(conn.export_code())
        print("WARNING - get_connection_genes: returned genes are single-connection only. Doesn't export influences.")
        return genes

    def is_hidden_node(self, idx):
        if (self.node_counts['input'] + self.node_counts['conduit']) <= idx < (self.node_counts['input'] + self.node_counts['conduit'] + self.node_counts['hidden']):
            return True
        else:
            return False

    def is_output_node(self, idx):
        if idx >= (self.node_counts['total'] - self.node_counts['output']):
            return True
        else:
            return False

    def is_input_node(self, idx):
        if idx < self.node_counts['input']:
            return True
        else:
            return False

    def is_conduit_neuron(self, idx):
        if self.node_counts['input'] <= idx < (self.node_counts['input'] + self.node_counts['conduit']):
            return True
        return False

    def summarize_connections(self, print_log=False):

        out_str = ""
        headers = "[input] , [conduit], [hidden] , [output]"
        node_idx = ("[{}-{}] , [{}-{}], [{}-{}] , [{}-{}]".format(0, self.node_counts['input'] - 1,
                                                                  self.node_counts['input'],
                                                                  self.node_counts['input'] + self.node_counts['conduit'] - 1,
                                                                  self.node_counts['input'] + self.node_counts['conduit'],
                                                                  self.node_counts['input'] + self.node_counts['conduit'] + self.node_counts['hidden'] - 1,
                                                                  self.node_counts['input'] + self.node_counts['conduit'] + self.node_counts['hidden'],
                                                                  self.node_counts['input'] + self.node_counts['conduit'] + self.node_counts['hidden'] + self.node_counts['output']))
        out_str += headers + "\n" + node_idx + "\n"
        if print_log: print(headers)
        if print_log: print(node_idx)

        for n, conn in enumerate(self.connections):
            conn_str = "conn {}: {} -> {}  | weight = {}, bias = {}  (active: {})".format(n, conn.src_node,
                                                                                          conn.dst_node, conn.weight,
                                                                                          conn.bias, conn.active)
            if print_log: print(conn_str)
            out_str += conn_str + "\n"

        return out_str

    def draw_graph(self, static_input_names=None, output_names=None, save_path=None, node_input_vals=None,
                   node_output_vals=None, sort_node_idx=False, rank_order='UD', data_input_count=None,
                   data_input_names=None, position_density_names=None):

        # ignore nodes/connections which don't lead to an output
        additional_path_to_output = True
        node_active = np.zeros(self.node_counts['total'])
        node_active[(self.node_counts['total']-self.node_counts['output']):] = 1
        connection_counted = np.zeros(len(self.connections))

        if output_names is None:
            output_names = []
            for i in range(self.node_counts['output']):
                output_names.append("Out-{}".format(i))
        if static_input_names is None:
            static_input_names = []
            for i in range(self.node_counts['input']):
                static_input_names.append("In-{}".format(i))

        # Iterate across all connections repeatedly. When a node is connected to an output, set
        # "additional_path_to_output" to True to ensure another loop. Set this source node to "active node" so that any
        # other connections with this node as a destination are also considered connected to an output.
        while additional_path_to_output:
            additional_path_to_output = False
            for i, conn in enumerate(self.connections):
                if not connection_counted[i] == 1:
                    if node_active[conn.dst_node] == 1:
                        node_active[conn.src_node] = 1
                        additional_path_to_output = True
                        connection_counted[i] = 1
        # "node_active" for each node indicates if that node is connected directly/indirectly to an output node and
        # should be displayed

        edges = []
        weights = []
        biases = []
        for conn in self.connections:
            if node_active[conn.dst_node] == 1:
                edges.append((conn.src_node, conn.dst_node))
                weights.append(conn.weight)
                biases.append(conn.bias)

        node_names = []
        node_types = []
        if node_output_vals is not None and node_input_vals is not None:
            node_vals = []
        else:
            node_vals = None
        for i, name in enumerate(static_input_names):
            node_names.append(name)
            node_types.append('I')
            if node_input_vals is not None:
                node_vals.append(node_input_vals[i])
        # If model contains data inputs, assigned data input nodes
        if data_input_count is not None:
            for i in range(data_input_count):
                if data_input_names is None:
                    node_names.append("data {}".format(i))
                else:
                    node_names.append(data_input_names[i])
                node_types.append('I')
                if node_input_vals is not None:
                    node_vals.append(node_input_vals[len(static_input_names) + i])
        # If using position density, assign input node names
        if position_density_names is not None:
            for i in range(len(position_density_names)):
                node_names.append(position_density_names[i])
                node_types.append('I')
                if node_input_vals is not None:
                    node_vals.append(node_input_vals[len(static_input_names) + data_input_count + i])
        # Conduit neurons
        for node in range(self.node_counts['input'], self.node_counts['input'] + self.node_counts['conduit']):
            node_names.append("C" + str(node - self.node_counts['input'] + 1))
            node_types.append('C')
            if node_output_vals is not None:
                node_vals.append(node_output_vals[node])
        # Hidden neurons
        for node in range(self.node_counts['input'] + self.node_counts['conduit'],
                          self.node_counts['input'] + self.node_counts['conduit'] + self.node_counts['hidden']):
            node_names.append("H" + str(node - (self.node_counts['input'] + self.node_counts['conduit']) + 1))
            node_types.append('H')
            if node_output_vals is not None:
                node_vals.append(node_output_vals[node])
        # Memory neurons
        for node in range(self.node_counts['input'] + self.node_counts['conduit'] + self.node_counts['hidden'],
                          self.node_counts['input'] + self.node_counts['conduit'] + self.node_counts['hidden'] + self.node_counts['memory']):
            node_names.append("M" + str(node - (self.node_counts['input'] + self.node_counts['conduit'] + self.node_counts['hidden']) + 1))
            node_types.append('M')
            if node_output_vals is not None:
                node_vals.append(node_output_vals[node])
        # Output neurons
        for i, name in enumerate(output_names):
            node_names.append(name)
            node_types.append('O')
            if node_output_vals is not None:
                node_vals.append(node_output_vals[self.node_counts['input'] + self.node_counts['conduit'] + self.node_counts['hidden'] + i])

        graph_img = generate_graph(node_names=node_names, node_types=node_types, edges=edges,
                                   weights=weights, biases=biases, node_vals=node_vals, sort_node_idx=sort_node_idx,
                                   rank_order=rank_order)

        if save_path is not None:
            import cv2
            cv2.imwrite(save_path, graph_img)
        return graph_img


class Connection:
    def __init__(self, src_node=None, dst_node=None, weight=0, bias=0, code=None, use_bias=True):
        super(Connection, self).__init__()

        self.use_bias = use_bias
        if code is None:
            self.weight = weight
            self.bias = bias
            self.src_node = src_node
            self.dst_node = dst_node
            self.code = self.export_code()
        elif code is not None:
            self.code = code
            self.import_code(code)
        else:
            print("ERROR: neither nodes nor code defined in Connection initialization")
            self.weight = weight
            self.bias = bias
            self.src_node = 0
            self.dst_node = 0
        self.active = True

    def calculate(self, input):
        if not self.use_bias:
            return input * self.weight
        else:
            return input * self.weight + self.bias

    def import_code(self, code):
        if not code[0] == '0':
            print("Error: Attempted to import non connection gene as connection")
            return
        self.src_node, self.dst_node, self.weight, self.bias, _ = decode_connection_gene(code)

    def export_code(self):
        return encode_connection_gene(self.src_node, self.dst_node, self.weight, self.bias)

    def summarize(self):
        summarize_connection_gene(encode_connection_gene(self.src_node, self.dst_node, self.weight, self.bias))


if __name__ == "__main__":
    '''
    connected gene format (binary):

    [src node] [dst node] [weight] [ bias ]
    [ 8-bit  ] [ 8-bit  ] [16-bit] [16-bit]
    total: 48 bits
    or 
    12 hex digits
    '''

    # Test Gene export:
    code = encode_connection_gene(src_node=0, dst_node=2, weight=0, bias=4)
    print("Decoded gene code:")
    print(decode_connection_gene(code))

    node_counts = {"input": 16, "output":8, "hidden":6, "conduit":5}
    node_counts["total"] = np.array([node_counts[k] for k in node_counts.keys()]).sum()

    # Test connection
    np.random.seed(12)
    code = generate_random_connection_gene(node_counts=node_counts)
    print(10 * "*")
    conn = Connection(code=code)
    print("src node: {}".format(conn.src_node))
    print("dst node: {}".format(conn.dst_node))
    print("weight: {}".format(conn.weight))
    print("bias: {}".format(conn.bias))

    # Test Brain with populated connections
    brain = Brain(node_counts=node_counts, max_connections=10)
    brain.random_init_connections()
    brain.Prune_Connections()

    for g in brain.get_connection_genes():
        summarize_connection_gene(g, input_names=['i1', 'i2', 'i3', 'i4'],
                                  output_names=['o1', 'o2', 'o3', 'o4', 'o5', 'o6'], internal_node_count=4, conduit_node_count=4,
                                  print_summary=True)

    print("[input] , [hidden] , [output]")
    print("[{}-{}] , [{}-{}] , [{}-{}]".format(0, 4, 5, 8, 9, 14))
    for n, conn in enumerate(brain.connections):
        print("conn {}: {} -> {}  ({})".format(n, conn.src_node, conn.dst_node, conn.active))

    print(brain.forward([1, 1, 3, 1, 2]))
