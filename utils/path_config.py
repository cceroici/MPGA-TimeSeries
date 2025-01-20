import os

class PathConfig:
    def __init__(self, sim_name, root_dir=None, populations=None, tensorboard_run_subdir=""):
        super(PathConfig, self).__init__()

        if root_dir is None:
            self.root_dir = os.getcwd().split("Genetic_Sim")[0] + "Genetic_Sim/"
        else:
            self.root_dir = root_dir
        self.sim_name = sim_name
        self.output_dir = self.root_dir + "/samples"
        self.samples_dir = self.output_dir + "/{}".format(sim_name)
        self.anim_preview_dir = self.output_dir + "/{}/anims/".format(sim_name)
        self.val_anim_preview_dir = self.output_dir + "/{}/val_anims/".format(sim_name)
        self.graph_preview_dir = self.output_dir + "/{}/graphs/".format(sim_name)
        self.debug_graph_dir = self.output_dir + "/{}/debug/".format(sim_name)
        self.log_dir = self.output_dir + "/{}/logs/".format(sim_name)
        self.plot_dir = self.output_dir + "/{}/plots/".format(sim_name)
        self.gene_pool_dir = self.output_dir + "/{}/gene_pool/".format(sim_name)
        self.tensorboard = self.root_dir + "/runs/{}".format(tensorboard_run_subdir)
        self.gs_state_dir = None
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(self.output_dir + "/{}".format(sim_name)):
            os.mkdir(self.output_dir + "/{}".format(sim_name))
        if not os.path.exists(self.val_anim_preview_dir):
            os.mkdir(self.val_anim_preview_dir)
        if not os.path.exists(self.anim_preview_dir):
            os.mkdir(self.anim_preview_dir)
        if not os.path.exists(self.graph_preview_dir):
            os.mkdir(self.graph_preview_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)
        if not os.path.exists(self.gene_pool_dir):
            os.mkdir(self.gene_pool_dir)
        if not os.path.exists(self.debug_graph_dir):
            os.mkdir(self.debug_graph_dir)

        # Generate paths for saving simulation/population state
        if not os.path.exists(self.root_dir + "/trained_states"):
            os.mkdir(root_dir + "/trained_states")
        self.gs_state_dir = self.root_dir + "/trained_states/" + sim_name + "/"
        if not os.path.exists(self.gs_state_dir):
            os.mkdir(self.gs_state_dir)


        # Generate population-specific paths
        self.pop_paths = None
        if populations is not None:
            if len(populations) > 1:
                self.pop_paths = []
                for i, population in enumerate(populations):
                    self.pop_paths.append(PopulationPaths(i, population.pop_label, self.anim_preview_dir,
                                                          self.graph_preview_dir, self.log_dir, self.plot_dir))


class PopulationPaths():
    def __init__(self, population_idx, population_label, anim_preview_dir, graph_dir, log_dir, plots_dir):
        super(PopulationPaths, self).__init__()

        population_str = "{} - {}".format(population_idx, population_label)

        # Path to population-specific /anims folder
        self.anim_dir = anim_preview_dir + "/" + population_str + "/"

        # Path to population-specific /graphs folder
        self.graph_dir = graph_dir + "/" + population_str + "/"

        # Path to population-specific /logs folder
        self.log_dir = log_dir + "/" + population_str + "/"

        # Path to population-specific /logs folder
        self.plot_dir = plots_dir + "/" + population_str + "/"

        if not os.path.exists(self.anim_dir):
            os.mkdir(self.anim_dir)
        if not os.path.exists(self.graph_dir):
            os.mkdir(self.graph_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)



