import os
import sys

if os.path.exists("C:/Users/ccero/Documents/Python/Genetic_Sim"):
    sys.path.append("C:/Users/ccero/Documents/Python/Genetic_Sim")
import numpy as np
import argparse
import json
from GeneticSim import Genetic_Sim_Parameters, GeneticSim
from src.Species import PopulationConfig
from utils.path_config import PathConfig
from species_config.species_gamer import Initialize_Gamer
from func_timeout import FunctionTimedOut, func_timeout
from src.game_config import tictactoe_config


def get_args():
    parser = argparse.ArgumentParser(description='TicTacToe')
    parser.add_argument("--no_generations", help='Number of training generations', type=int, default=5001)
    parser.add_argument("--simulation_name", help='Name for saving model/samples', type=str, default="tictactoe_sim")
    parser.add_argument("--batch_mode", help="If \"True\" run in batch mode", type=str, default="False")
    parser.add_argument("--batch_run_name", help="Name of batch run", type=str, default="")
    parser.add_argument("--mutation_rate", help="(float) decimal probability of mutation (default: 0.01)", type=float, default=1e-2)
    parser.add_argument("--gene_count", help="(float) number of genes per organism (default: 32)", type=int, default=32)
    parser.add_argument("--population_size", help="(int) number of organisms per population (default: 750)", type=int, default=750)
    parser.add_argument("--full_validation_only", help="If 'True', run full validation instead of training", type=str, default="False")
    parser.add_argument("--root_dir", help="Path to root location to look for trained models and save samples")
    parser.add_argument("--report_freq", help="Number of generations between generating summary figures and saving checkpoint")
    return parser.parse_args()


def run(args, sim_name=None, data_file_name=None, DEBUG_MODE=False, batch_mode=False, batch_name=None,
        load_genome=False):

    if sim_name is None:
        sim_name = args.simulation_name

    report_freq = 500
    if args.report_freq is not None:
        report_freq = int(args.report_freq)

    root_dir = "./"
    if args.root_dir is not None:
        root_dir = args.root_dir

    print(sim_name)

    winner_config = PopulationConfig(pop_count=args.population_size, genes_per_organism=args.gene_count, internal_neuron_count=16,
                                     conduit_neuron_count=2, memory_neuron_count=0,
                                     decision_threshold=0.95, max_health=10000, min_health=-10000, start_health=50.,
                                     fitness_type='ROI-numerical', max_CLK_period=500, max_RR_period=800,
                                     max_connections=500,
                                     pher_channel=0, PD_channel=0,
                                     use_spread=True, use_trans_fee=True,
                                     pop_label="W", fan_color=(0, 0, 1), line_color=(0, 0.5, .7))

    '''    loser_config = PopulationConfig(pop_count=args.population_count, genes_per_organism=args.gene_count, internal_neuron_count=16,
                                        conduit_neuron_count=16, memory_neuron_count=8,
                                        decision_threshold=0.95, max_health=10000, min_health=-10000, start_health=50.,
                                        fitness_type='ROI-numerical-loser', max_CLK_period=500, max_RR_period=800,
                                        pher_channel=1, PD_channel=1,
                                        max_connections=500,
                                        use_spread=False, use_trans_fee=False,
                                        pop_label="L", fan_color=(1, .4, 0), line_color=(.8, 0.6, 0))
    
        winner_config_nospread = PopulationConfig(pop_count=args.population_count, genes_per_organism=args.gene_count,
                                                  internal_neuron_count=16, conduit_neuron_count=16, memory_neuron_count=8,
                                                  decision_threshold=0.95, max_health=10000, min_health=-10000,
                                                  start_health=50.,
                                                  fitness_type='ROI-numerical', max_CLK_period=500, max_RR_period=800,
                                                  pher_channel=2, PD_channel=2,
                                                  max_connections=500,
                                                  use_spread=False, use_trans_fee=False,
                                                  pop_label="S", fan_color=(0, .8, .2), line_color=(0, .7, .2))'''

    #species_configs = [winner_config, loser_config, winner_config_nospread]
    species_configs = [winner_config]
    print("Training with {} populations".format(len(species_configs)))

    tictactoe_game = tictactoe_config()
    gamer = Initialize_Gamer(game_config=tictactoe_game, no_populations=len(species_configs),
                             memory_node_count=winner_config.memory_neuron_count)
    species_templates = [gamer for p in species_configs]

    path_config = PathConfig(sim_name=sim_name, populations=species_configs, root_dir=root_dir,
                             tensorboard_run_subdir="" if batch_mode is False else batch_name)
    # path_config.gs_state_dir = "./trained_states/crypto-3min-intraday-btc_base-RR-GPU_CPU_check-1001/"
    print("Saving samples to \"{}\"".format(path_config.output_dir))

    genetic_params = Genetic_Sim_Parameters(mutation_rate=args.mutation_rate, crossover_rate=0.5, tournament_size=2,
                                            elite_count=2,
                                            gene_dupl_rate=0, gene_skip_rate=0)

    Gsim = GeneticSim(species_templates=species_templates, species_configurations=species_configs,
                      genetic_sim_params=genetic_params, time_steps_per_gen=None, position_count=tictactoe_game.no_decisions,
                      path_config=path_config,
                      preview_interval=report_freq, test_interval=report_freq, sim_type="game",
                      load_genome_path=None, game_config=tictactoe_game)

    if not Gsim.initialized:
        print("ERROR: Failed to initialize GeneticSim()")
        while True:
            pass

    from Simulations.Games.TicTacToe_utils import get_designer_genes
    designer_genes = get_designer_genes(Gsim.populations[0].genes_per_organism,
                                        output_offset=Gsim.populations[0].out_offset)

    for i, o in enumerate(Gsim.populations[0].organisms):
        if i>-1:
            break
        if i==0:
            print(10*"*" + " USING DESIGNER GENES")
        o.brain.connections = []
        o.load_genes(designer_genes)

    Gsim.train(no_generations=args.no_generations, skip_animation=True)
    np.random.seed(555)
    Gsim.Final_Validation(time_steps=None, USE_GPU=True, FULL_VALIDATION=False,
                          SAVE_METRICS=True, validation_dataset_name=data_file_name)

    run_description = ""
    run_metadata = {"description": run_description, "training dataset": data_file_name}
    with open(path_config.gs_state_dir + "training_run_metadata.json", "w") as write_file:
        json.dump(run_metadata, write_file, indent=4)

        Gsim.save_genome(path_config.gs_state_dir + "GS_state.p", path_config.gs_state_dir + "config.json",
                         args.no_generations)
        Gsim.save_genome(path_config.gene_pool_dir + "GS_state.p", path_config.gene_pool_dir + "config.json",
                         args.no_generations)



if __name__ == "__main__":
    np.random.seed(1)

    import sys

    # 'True' when debugging
    DEBUG_MODE = '_pydev_bundle.pydev_log' in sys.modules.keys()

    args = get_args()
    run(args=args, DEBUG_MODE=DEBUG_MODE)
