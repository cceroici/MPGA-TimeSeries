import numpy as np
from src.genome import random_splice_genes, apply_mutation
from src.species import Organism_Numerical
from src.selection import tournament_selection
from src.environment import initial_positions

"""
Code for performing:
 - Organism selection
 - Gene recombination/mutation
 - New population generation
"""

'''
def Repopulate_From_Gene_Pool(pop_count, gene_pool, genes_per_organism, initial_pos_sigma, sim_size):

    organisms = []
    org_rand_pos = Initial_Positions(N=pop_count, sigma=initial_pos_sigma, sim_size=sim_size)
    rand_gene_idx = np.random.randint(low=0, high=len(gene_pool), size=(pop_count, genes_per_organism))
    for i in range(pop_count):
        new_org = Organism_2D(org_rand_pos[i, :], gene_count=organisms[0].gene_count,
                             input_count=organisms[0].brain.input_count,
                             output_count=organisms[0].brain.output_count,
                             internal_neuron_count=organisms[0].brain.internal_neuron_count,
                             max_connections=organisms[0].brain.max_connections)
        new_org.load_genes(gene_pool[rand_gene_idx[i, :]])
        organisms.append(new_org)
    return organisms'''


def tournament_repopulate(mutation_rate, crossover_rate, population, initial_pos_sigma,
                          sim_size, tournament_size, elite_count=0, skip_rate=0.0, dupl_rate=0.0,
                          sim_type='2D', position_count=None, profiler=None, tie_breaker=None):
    if isinstance(population.fitness, str):
        if population.fitness == 'age':
            age = []
            for org in population.organisms:
                age.append(org.age)
            age = np.array(age)
            population.fitness = age
    #profiler.start_measurement("init positions")
    org_rand_pos = initial_positions(no_org=population.pop_count, sigma=initial_pos_sigma, sim_size=sim_size,
                                     position_count=position_count,
                                     distribution_type='Gaussian' if sim_type == '2D' else 'numerical-binary')
    #profiler.end_measurement("init positions")

    organisms_new = []
    gene_pool = []

    # capture elite organisms
    #profiler.start_measurement("elite")
    if population.fitness_cap is not None:
        population.fitness[population.fitness>population.fitness_cap] = population.fitness_cap
    if elite_count > 0:
        fitness_sort_idx = np.argsort(population.fitness)
        for i in range(1, elite_count + 1):
            elite_genes = population.organisms[fitness_sort_idx[-i]].genes
            elite_child = Organism_Numerical(org_rand_pos[i, :], gene_count=population.genes_per_organism,
                                             input_count=population.node_counts["input"],
                                             output_count=population.node_counts["output"],
                                             internal_neuron_count=population.node_counts["hidden"],
                                             conduit_neuron_count=population.node_counts["conduit"],
                                             memory_neuron_count=population.node_counts["memory"],
                                             max_connections=population.max_connections)
            elite_child.load_genes(elite_genes)
            organisms_new.append(elite_child)
    #profiler.end_measurement("elite")

    n = elite_count
    sel_count = np.zeros(population.pop_count)
    while len(organisms_new) < population.pop_count:
        #profiler.start_measurement("tourny select")

        parent1_idx = tournament_selection(fitness=population.fitness, group_size=tournament_size, tie_breaker=tie_breaker)
        parent2_idx = tournament_selection(fitness=population.fitness, group_size=tournament_size, tie_breaker=tie_breaker)
        #profiler.end_measurement("tourny select")

        sel_count[parent1_idx] += 1
        sel_count[parent2_idx] += 1

        rnd = np.random.rand(2 * (len(population.organisms[0].genes) + 1))
        rnd_state = np.random.get_state()
        #profiler.start_measurement("splice")

        child1_genes = random_splice_genes(
            genomes=[population.organisms[parent1_idx].genes, population.organisms[parent2_idx].genes],
            crossover_rate=crossover_rate, skip_rate=skip_rate, dupl_rate=dupl_rate,
            rnd_array=rnd, rnd_idx=0)
        child2_genes = random_splice_genes(
            genomes=[population.organisms[parent1_idx].genes, population.organisms[parent2_idx].genes],
            crossover_rate=crossover_rate, skip_rate=skip_rate, dupl_rate=dupl_rate,
            rnd_array=rnd, rnd_idx=len(population.organisms[0].genes))
        #profiler.end_measurement("splice")

        #profiler.start_measurement("mutate")

        np.random.set_state(rnd_state)
        for i in range(len(child1_genes)):
            r_mut = np.random.rand()
            while r_mut < mutation_rate:

                child1_genes[i] = apply_mutation(child1_genes[i], node_counts=population.organisms[0].brain.node_counts,
                                                 max_CLK_lim=population.max_CLK_period,
                                                 max_RR_lim=population.max_RR_period)
                r_mut = np.random.rand()
            r_mut = np.random.rand()
            while r_mut < mutation_rate:
                child2_genes[i] = apply_mutation(child2_genes[i], node_counts=population.organisms[0].brain.node_counts,
                                                 max_CLK_lim=population.max_CLK_period,
                                                 max_RR_lim=population.max_RR_period)
                r_mut = np.random.rand()
        #profiler.end_measurement("mutate")

        gene_pool += child1_genes
        gene_pool += child2_genes

        child1 = Organism_Numerical(org_rand_pos[n, :], gene_count=population.genes_per_organism,
                                    input_count=population.node_counts["input"],
                                    output_count=population.node_counts["output"],
                                    internal_neuron_count=population.node_counts["hidden"],
                                    conduit_neuron_count=population.node_counts["conduit"],
                                    memory_neuron_count=population.node_counts["memory"],
                                    max_connections=population.max_connections)
        child2 = Organism_Numerical(org_rand_pos[n + 1, :], gene_count=population.genes_per_organism,
                                    input_count=population.node_counts["input"],
                                    output_count=population.node_counts["output"],
                                    internal_neuron_count=population.node_counts["hidden"],
                                    conduit_neuron_count=population.node_counts["conduit"],
                                    memory_neuron_count=population.node_counts["memory"],
                                    max_connections=population.max_connections)
        #profiler.start_measurement("load genes")

        child1.load_genes(child1_genes)
        child2.load_genes(child2_genes)
        #profiler.end_measurement("load genes")

        organisms_new.append(child1)
        organisms_new.append(child2)
        n += 1

    if len(organisms_new) > population.pop_count:
        organisms_new = organisms_new[0:population.pop_count]

    population.organisms = organisms_new

    return gene_pool, sel_count
