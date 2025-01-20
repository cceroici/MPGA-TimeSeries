import pickle
from src.species import Species


def Initialize_Gamer(game_config, no_populations=1, memory_node_count=0):
    """
    Specify fixed inputs/outputs
    data input start indexes defined here, but names/labels are stored in environment
    """
    gamer = Species()
    gamer.game_input_count = game_config.no_inputs
    gamer.game_decision_count = game_config.no_decisions
    gamer.data_input_count = game_config.no_inputs
    gamer.position_names = game_config.decision_names

    gamer.input_names = []
    gamer.output_names = []

    # indexes to inputs/outputs by name
    gamer.in_idx = dict()
    gamer.out_idx = dict()

    # Populate inputs - organism state inputs (last decision vector for specific organism) [no_positions]
    for i, inp_name in enumerate(game_config.input_names):
        gamer.input_names.append(inp_name)
        gamer.in_idx[inp_name] = i

    # Populate outputs game decisions
    for i, dec_name in enumerate(game_config.decision_names):
        gamer.output_names.append(dec_name)
        gamer.out_idx[dec_name] = i

    gamer.input_names.append("actb")
    gamer.in_idx["actb"] = len(gamer.in_idx)

    # Fixed outputs

    # Memory node store outputs
    for i in range(memory_node_count):
        gamer.output_names.append("MEM{}".format(i))
        gamer.out_idx["MEM store {}".format(i)] = len(gamer.out_idx)

    gamer.Initialize(last_fixed_input_idx=gamer.in_idx["actb"], no_pos_density_ch=no_populations,
                     position_count=gamer.game_decision_count)

    return gamer


if __name__ == "__main__":

    from src.game_config import tictactoe_config

    spec_config = Initialize_Gamer(game_config=tictactoe_config(), no_populations=2, memory_node_count=2)

    a=1