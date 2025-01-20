import cv2
import numpy as np
from PIL import Image
import io
from matplotlib import pyplot as plt
try:
    from graphviz import Digraph
    import networkx as nx
    from networkx.drawing.nx_agraph import to_agraph
    GRAPHVIS = True
except:
    print("Warning: networkx not installed. Skipping all graph visualizations.")
    GRAPHVIS = False

def generate_graph(node_names, node_types, edges, weights, biases, show_bias_label=False, node_vals=None,
                   show_all_nodes=False, sort_node_idx=False, rank_order='UD'):

    if not GRAPHVIS:
        return np.zeros((512, 512, 3), dtype="uint8")

    nodes = []
    if show_all_nodes:
        for i in range(len(node_types)):
            nodes.append(i)
    else:
        for edge in edges:
            if edge[0] not in nodes:
                nodes.append(edge[0])
            if edge[1] not in nodes:
                nodes.append(edge[1])
    nodes = np.array(nodes)

    # sort node idx so inputs are first, outputs are last
    if sort_node_idx:
        input_nodes = []
        conduit_nodes = []
        hidden_nodes = []
        output_nodes = []
        for node in nodes:
            if node_types[node] == "I":
                input_nodes.append(node)
            elif node_types[node] == "C":
                conduit_nodes.append(node)
            elif node_types[node] == "H":
                hidden_nodes.append(node)
            elif node_types[node] == "O":
                output_nodes.append(node)
        input_nodes.sort()
        conduit_nodes.sort()
        hidden_nodes.sort()
        output_nodes.sort()
        nodes = np.array(input_nodes + conduit_nodes + hidden_nodes + output_nodes)


    node_bias = np.zeros(len(nodes))
    for i, edge in enumerate(edges):
        node_bias[np.argwhere(nodes == edge[1])[0][0]] += biases[i]

    for i, bias in enumerate(node_bias):
        if np.abs(bias) > 4:
            node_bias[i] = bias / np.abs(bias) * 4

    g = Digraph()
    g.attr(rankdir=rank_order, size='24,24')

    for i, node in enumerate(nodes):
        if node_types[node] == "I":  # inputs
            if node_vals is None:
                fillcolor = '#c9ffe6'
                if node_names[node] in ["CLK", 'age']:
                    fillcolor = '#fffbc9'
                elif node_names[node] in ["pop-dens"]:
                    fillcolor = '#ffd0ff'
            else:
                if node_vals[node] < 0:
                    c_node = np.array([255, 255 + 255 * node_vals[node], 255 + 255 * node_vals[node]]).astype('int')
                else:
                    c_node = np.array([255 - 255 * node_vals[node], 255, 255 - 255 * node_vals[node]]).astype('int')
                c_node[c_node > 255] = 255
                c_node[c_node < 0] = 0
                fillcolor = "#{:02x}{:02x}{:02x}".format(c_node[0], c_node[1], c_node[2])

            g.attr('node', shape='doublecircle', style='filled', color='#2E8B57', penwidth='2', fillcolor=fillcolor,
                   fontcolor='#000000')
        elif node_types[node] == "O":  # outputs
            if node_vals is None:
                fillcolor = '#ffd5a6'
            else:

                c_node = np.array([235, 235, 255 - 255 * node_vals[node]]).astype('int')
                c_node[c_node > 255] = 255
                fillcolor = "#{:02X}{:02X}{:02X}".format(c_node[0], c_node[1], c_node[2])
            g.attr('node', shape='box', style='filled', color='#0000ff' if node_bias[i] < 0 else '#ff0000',
                   penwidth='{}'.format(np.abs(node_bias[i])), fillcolor=fillcolor, fontcolor='#000000')
        elif node_types[node] == "C":  # 'C': conduit
            if node_vals is None:
                fillcolor = '#50FFFF'
                fontcolor = '#000000'
            else:
                c_node = np.array([255, 255, 255])
                if node_vals[node] < 0:
                    c_node[2] = c_node[2] - 255 * (node_vals[node])
                    c_node[1] = c_node[1] - 255 * (node_vals[node])
                    c_node[0] = c_node[0] + (255-200)*(node_vals[node])
                else:
                    c_node = c_node - (255-60)*(node_vals[node])
                    c_node = c_node.astype('int')
                if np.mean(c_node) > 175:
                    fontcolor = '#000000'
                else:
                    fontcolor = '#FFFFFF'
                c_node -= 20
                c_node[c_node > 255] = 255
                c_node[c_node < 0] = 0

                fillcolor = "#{:02x}{:02x}{:02x}".format(c_node[0], c_node[1], c_node[2])
            g.attr('node', shape='circle', style='filled', color='#0000ff' if node_bias[i] < 0 else '#ff0000',
                   penwidth='{}'.format(np.abs(node_bias[i])), fillcolor=fillcolor, fontcolor=fontcolor)
        elif node_types[node] == "H":  # 'H': internal/hidden
            if node_vals is None:
                fillcolor = '#3b3b3b'
                fontcolor = '#FFFFFF'
            else:
                c_node = np.array([255, 255, 255])
                if node_vals[node] < 0:
                    c_node[2] = c_node[2] - 255 * (node_vals[node])
                    c_node[1] = c_node[1] - 255 * (node_vals[node])
                    c_node[0] = c_node[0] + (255-200)*(node_vals[node])
                else:
                    c_node = c_node - (255-60)*(node_vals[node])
                    c_node = c_node.astype('int')
                if np.mean(c_node) > 175:
                    fontcolor = '#000000'
                else:
                    fontcolor = '#FFFFFF'
                c_node -= 20
                c_node[c_node > 255] = 255
                c_node[c_node < 0] = 0

                fillcolor = "#{:02x}{:02x}{:02x}".format(c_node[0], c_node[1], c_node[2])
            g.attr('node', shape='circle', style='filled', color='#0000ff' if node_bias[i] < 0 else '#ff0000',
                   penwidth='{}'.format(np.abs(node_bias[i])), fillcolor=fillcolor, fontcolor=fontcolor)
        g.node(node_names[node])

    for i, edge in enumerate(edges):
        if show_bias_label:
            g.attr('edge', color='blue' if weights[i] > 0 else 'red', penwidth='{}'.format(np.abs(weights[i])),
                   label="{:0.1f}".format(biases[i]))
        else:
            g.attr('edge', color='blue' if weights[i] > 0 else 'red', penwidth='{}'.format(np.abs(weights[i])))
        g.edge(node_names[edge[0]], node_names[edge[1]], label="{:0.1f}".format(biases[i]))

    img_data = g.pipe(format='png')
    img = Image.open(io.BytesIO(img_data))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return img


if __name__ == "__main__":
    node_names = ['A', 'B', 'C', 'D', 'E']
    node_types = ['I', 'H', 'C', 'O', 'H']
    edges = [(0, 1), (0, 2), (4, 1), (4, 4), (0, 3)]
    weights = [0.1, 1, 1.5, -0.5, 0.4]
    biases = [0.1, -10, -0.5, 0.7, 0.5]
    node_values = [-0.5, 1, 0.5, 0, -0.5]
    node_values = None
    graph = generate_graph(node_names=node_names, node_types=node_types, edges=edges, weights=weights, biases=biases,
                           node_vals=node_values)
    plt.imshow(graph)
    plt.show()
