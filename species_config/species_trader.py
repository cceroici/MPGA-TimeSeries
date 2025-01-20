import pickle
from src.species import Species


def initialize_trader(memory_node_count, no_populations, pickle_data_file=None, no_data_inputs=None, currencies=None):
    """
    Specify fixed inputs/outputs
    data input start indexes defined here, but names/labels are stored in environment
    """
    if pickle_data_file is not None:
        env_data_file = pickle.load(open(pickle_data_file, "rb"))
        currencies = env_data_file["currencies"]
        data_inputs = env_data_file["input data"]
        no_data_inputs = len(data_inputs)

    trader = Species()

    trader.position_count = len(currencies)
    trader.data_input_count = no_data_inputs
    trader.position_names = currencies
    trader.input_names = []
    trader.output_names = []

    # indexes to inputs/outputs by name
    trader.in_idx = dict()
    trader.out_idx = dict()

    # Populate inputs/outputs with specified currencies
    for i, currency in enumerate(currencies):
        trader.input_names.append("pos-" + currency)
        trader.output_names.append(currency)
        trader.in_idx["pos-" + currency] = i
        trader.out_idx[currency] = i

    # Fixed inputs
    trader.input_names.append("CLK")
    trader.input_names.append("age")
    trader.in_idx["CLK"] = len(trader.in_idx)
    trader.in_idx["age"] = len(trader.in_idx)

    # Fixed outputs
    trader.output_names.append("CLK-P+")
    trader.output_names.append("CLK-P-")
    trader.output_names.append("RR-P+")
    trader.output_names.append("RR-P-")
    trader.output_names.append("RR-OV")
    trader.output_names.append("THRESH+")
    trader.output_names.append("THRESH-")
    trader.out_idx["CLK limit +"] = len(trader.out_idx)
    trader.out_idx["CLK limit -"] = len(trader.out_idx)
    trader.out_idx["RR limit +"] = len(trader.out_idx)
    trader.out_idx["RR limit -"] = len(trader.out_idx)
    trader.out_idx["RR OV"] = len(trader.out_idx)
    trader.out_idx["THRESH +"] = len(trader.out_idx)
    trader.out_idx["THRESH -"] = len(trader.out_idx)


    # Memory node store outputs
    for i in range(memory_node_count):
        trader.output_names.append("MEM{}".format(i))
        trader.out_idx["MEM store {}".format(i)] = len(trader.out_idx) + i

    trader.Initialize(last_fixed_input_idx=trader.in_idx["age"], position_count=trader.position_count,
                      no_pos_density_ch=no_populations)

    return trader