import numpy as np
import argparse
import json
from genetic_sim import GeneticSimParameters, GeneticSim, load_trader_model
from src.species import PopulationConfig
from utils.path_config import PathConfig
from species_config.species_trader import initialize_trader


def get_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Crypto Trader')
    # Path to dataset containing training/validation data
    parser.add_argument("--dataset_file", help='Path to training dataset', type=str, default="datasets/crypto_dataset_3minute_2024.p")
    # Number of generations (epochs) for training
    parser.add_argument("--no_generations", help='Number of training generations', type=int, default=5001)
    # Label for saving results/checkpoints
    parser.add_argument("--simulation_name", help='Name for saving model/samples', type=str, default="crypto_trader")
    # Number of time steps for each generation
    parser.add_argument("--generation_duration", help='Number of iterations/steps per generation (integer)', type=int, default=500)
    # If 'True', the W population will include trading fees in ROI calculations
    parser.add_argument("--use_fee", help="If \"True\" the 'winner' population uses transaction fees", type=str2bool, default="True")
    # If 'True', the W population will include ask/bid spread in ROI calculations
    parser.add_argument("--use_spread", help="If \"True\" the 'winner' population uses transaction spread", type=str2bool, default="True")
    # Maximum number of sequential internal cascade nodes to compute for each time step
    parser.add_argument("--conduit_solve_depth", help="Number of iterations for evaluating conduit node paths", type=int, default=2)
    # Probability of gene mutation during repopulation
    parser.add_argument("--mutation_rate", help="(float) probability of mutation (default: 0.01)", type=float, default=1e-2)
    # Numer of genes per organism
    parser.add_argument("--gene_count", help="(float) number of genes per organism (default: 32)", type=int, default=32)
    # Number of organisms per population
    parser.add_argument("--population_count", help="(int) number of organisms per population (default: 750)", type=int, default=512)
    # Set to >0 to penalize populations for not trading anything (common behaviour in the W population due to fees)
    parser.add_argument("--no_trade_penalty", help="Loss penalty for not trading (default: 0.0)", type=float, default=0.)
    # Minimum number of trades per generation in order to not be penalized
    parser.add_argument("--no_trade_threshold", help="Trade count threshold for not trading", type=int, default=0)
    # Weight decay for regularization
    parser.add_argument("--weight_decay", help="Factor of weight decay", type=float, default=0.)
    # Probability of gene duplication during recombination
    parser.add_argument("--gene_dupl_rate", help="Gene duplication rate", type=float, default=0.)
    # Probability of gene skipping during recombination
    parser.add_argument("--gene_skip_rate", help="Gene duplication rate", type=float, default=0.)
    # Number of selected 'elites' during tournament selection
    parser.add_argument("--elite_count", help="Tournament repopulate elite count", type=int, default=3)
    # Number of competitors selected during the tournament selection
    parser.add_argument("--tournament_size", help="Tournament repopulate competition size", type=int, default=2)
    # Randomized dropout mask for improved regularization
    parser.add_argument("--dropout", help="Dropout factor", type=float, default=0.)
    # Spoiled population is the population index that all other populations are rewarded for improving. Usually would set this to be the W population.
    parser.add_argument("--spoiled_pop_idx", help="\'Spoiled\' population index", type=int, default=None)
    # Parameter to scale the benefit of the spoiled population to other populations
    parser.add_argument("--spoiled_pop_gamma", help="\'Spoiled\' population fitness factor", type=float, default=None)
    # Select training or validate or both (train+validate)
    parser.add_argument("--mode", help="For training, include 'train' in argument, for validation include 'validate' (eg. 'train+validate' for both)", type=str, default="train+validate")
    # Root directory to save results
    parser.add_argument("--root_dir", help="Path to root location to look for model checkpoints and save samples", default="./outputs")
    # Number of generations in between saving results/checkpoint
    parser.add_argument("--report_freq", help="Number of generations between generating summary figures and saving checkpoints", type=int, default=500)
    # Notes on run (optional)
    parser.add_argument("--run_description", help="High-level description of simulation", default="")
    # If resuming a previous checkpoint, this path points to the genome folder in the "trained_states" directory
    parser.add_argument("--load_model", help="'\True'\ to load model from the folder specified by '\simulation_name'\'", type=str2bool, default="False")

    return parser.parse_args()


def run(args, load_model=False):

    train_model = False
    validate_model = False
    if 'train' in args.mode:
        train_model = True
    if 'validate' in args.mode:
        validate_model = True

    environment_params = {"data type": "numerical-data-file",
                          "data file": args.dataset_file,
                          "data interval mode": "intraday"}

    genetic_params = GeneticSimParameters(mutation_rate=args.mutation_rate, crossover_rate=0.5, tournament_size=args.tournament_size,
                                          elite_count=args.elite_count,
                                          gene_dupl_rate=args.gene_dupl_rate, gene_skip_rate=args.gene_skip_rate,
                                          no_trade_penalty=args.no_trade_penalty,
                                          no_trade_threshold=args.no_trade_threshold,
                                          weight_decay_gamma=args.weight_decay, dropout=args.dropout,
                                          spoiled_pop_idx=args.spoiled_pop_idx,
                                          spoiled_pop_gamma=args.spoiled_pop_gamma)

    # Load model configuration and genome from saved model files
    if (validate_model and not train_model) or load_model:
        print("Loading saved model from: {}".format("./trained_states/" + args.sim_name))
        Gsim = load_trader_model(load_model_folder="./trained_states/" + args.sim_name, environment_params=environment_params,
                                 genetic_params=genetic_params)
    else:  # Train new model from scratch
        winner_config = PopulationConfig(pop_count=args.population_count, genes_per_organism=args.gene_count,
                                         internal_neuron_count=20, conduit_neuron_count=8, memory_neuron_count=2,
                                         population_node_count=4,
                                         decision_threshold=0.95, max_health=10000, min_health=-10000, start_health=50.,
                                         fitness_type='ROI-numerical', max_CLK_period=500, max_RR_period=800,
                                         max_connections=500,
                                         pher_channel=0, PD_channel=0,
                                         use_spread=args.use_spread, use_trans_fee=args.use_fee,
                                         pop_label="W", fan_color=(0, 0, 1), line_color=(0, 0.5, .7), fitness_cap=1.04)

        loser_config = PopulationConfig(pop_count=args.population_count, genes_per_organism=args.gene_count,
                                        internal_neuron_count=20, conduit_neuron_count=8, memory_neuron_count=2,
                                        population_node_count=4,
                                        decision_threshold=0.95, max_health=10000, min_health=-10000, start_health=50.,
                                        fitness_type='ROI-numerical-loser', max_CLK_period=500, max_RR_period=800,
                                        pher_channel=1, PD_channel=1,
                                        max_connections=500,
                                        use_spread=False, use_trans_fee=False,
                                        pop_label="L", fan_color=(1, .4, 0), line_color=(.8, 0.6, 0))

        winner_config_nospread = PopulationConfig(pop_count=args.population_count, genes_per_organism=args.gene_count,
                                                  internal_neuron_count=20, conduit_neuron_count=8, memory_neuron_count=2,
                                                  population_node_count=4,
                                                  decision_threshold=0.95, max_health=10000, min_health=-10000,
                                                  start_health=50.,
                                                  fitness_type='ROI-numerical', max_CLK_period=500, max_RR_period=800,
                                                  pher_channel=2, PD_channel=2,
                                                  max_connections=500,
                                                  use_spread=False, use_trans_fee=False,
                                                  pop_label="S", fan_color=(0, .8, .2), line_color=(0, .7, .2))

        species_configs = [winner_config, loser_config, winner_config_nospread]  # Use 3 populations
        #species_configs = [winner_config, loser_config]                         # Use 2 populations
        #species_configs = [winner_config]                                       # Use 1 popuation
        print("Training with {} populations".format(len(species_configs)))

        trader = initialize_trader(pickle_data_file=environment_params["data file"], no_populations=len(species_configs),
                                   memory_node_count=winner_config.memory_neuron_count)

        species_templates = [trader for p in species_configs]

        path_config = PathConfig(sim_name=args.simulation_name, populations=species_configs, root_dir=args.root_dir)
        print("Saving samples to \"{}\"".format(path_config.output_dir))

        Gsim = GeneticSim(species_templates=species_templates, species_configurations=species_configs,
                          genetic_sim_params=genetic_params, time_steps_per_gen=args.generation_duration,
                          position_count=trader.position_count,
                          path_config=path_config,
                          preview_interval=args.report_freq, test_interval=args.report_freq, sim_type="numerical",
                          load_genome_path=None,
                          environment_parameters=environment_params, conduit_solve_depth=args.conduit_solve_depth)

    if not Gsim.initialized:
        print("ERROR: Failed to initialize GeneticSim()")
        while True:
            pass

    try:
        from numba.cuda import detect
        detect()
        use_gpu = True
    except Exception as e:
        print("CUDA not detected. Using CPU.")
        use_gpu = False

    if train_model:
        Gsim.train(no_generations=args.no_generations, USE_GPU=use_gpu)
        np.random.seed(0)
        Gsim.Final_Validation(time_steps=None, USE_GPU=True, FULL_VALIDATION=False,
                              SAVE_METRICS=True, validation_dataset_name=args.dataset_file)
        if args.blind:
            val_date = Gsim.env.env_data_file['dates'][-1].isoformat()
        elif "validation date" in list(Gsim.env.env_data_file.keys()):
            val_date = Gsim.env.env_data_file['validation date'].isoformat()
        else:
            val_date = None
        run_metadata = {"description": args.run_description, "training dataset": args.dataset_file,
                        "dataset end date": val_date}
        with open(path_config.gs_state_dir + "training_run_metadata.json", "w") as write_file:
            json.dump(run_metadata, write_file, indent=4)

        Gsim.save_genome(path_config.gs_state_dir + "GS_state.p", path_config.gs_state_dir + "config.json",
                         args.no_generations)
        Gsim.save_genome(path_config.gene_pool_dir + "GS_state.p", path_config.gene_pool_dir + "config.json",
                         args.no_generations)

    if validate_model:
        if train_model:
            Gsim.visualizer.reset_population_data()
        Gsim.final_validation(time_steps=None, USE_GPU=True, FULL_VALIDATION=True,
                              SAVE_METRICS=True, validation_dataset_name=args.dataset_file,
                              visualization_config={"mode": "full", "plotly summary": True, "performance log": True})


if __name__ == "__main__":
    np.random.seed(0)
    args = get_args()
    run(args=args, load_model=args.load_model)

