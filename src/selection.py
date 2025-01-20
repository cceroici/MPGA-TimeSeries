import numpy as np


def apply_selection(organisms, selection_type):
    surviving_count = 0
    if selection_type == "corners":  # only organisms at the four corners survive at the end of each generation
        for org in organisms:
            if -30 < org.pos[0] < 30:
                org.alive = False
            elif -30 < org.pos[1] < 30:
                org.alive = False
            else:
                surviving_count += 1
        survival_rate = surviving_count / len(organisms)
        return organisms, survival_rate
    elif selection_type == "age":
        avg_age = get_average_age(organisms)
        return organisms, avg_age


def get_average_age(organisms, living_only=False):
    avg_age = 0
    count = 0
    for org in organisms:
        if org.alive or not living_only:
            avg_age += org.age
            count += 1
    if count == 0:
        return 0
    avg_age /= count
    return avg_age


def get_average_health(organisms, living_only=False):
    avg_health = 0
    count = 0
    for org in organisms:
        if org.alive or not living_only:
            avg_health += org.health
            count += 1
    if count == 0:
        return 0
    avg_health /= count
    return avg_health


def tournament_selection(fitness, group_size, tie_breaker=None):

    grp_idx = np.random.randint(low=0, high=len(fitness), size=group_size)

    winner_idx = grp_idx[np.argmax(fitness[grp_idx])]

    if tie_breaker is not None:
        tie = np.all(fitness[grp_idx] == fitness[grp_idx[0]])
        if tie:
            #    print("TIE: {} -> F{} --> TB{}".format(grp_idx, fitness[grp_idx], tie_breaker[grp_idx]))
            winner_idx = grp_idx[np.argmax(tie_breaker[grp_idx])]

    return winner_idx


def roulette_wheel():
    # Not implmeneted
    pass


if __name__ == "__main__":

    fitness = np.random.rand(10)
    tournament_selection(fitness=fitness, group_size=3)