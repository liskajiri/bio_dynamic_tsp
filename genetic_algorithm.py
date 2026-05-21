from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pygad
from loguru import logger

from data import DynamicRoadTSP, ExperimentConfig, setup_experiment
from utils import tour_to_route_nodes


@dataclass
class PygadConfig:
    population_size: int = 200
    iterations: int = 1000
    num_parents_mating: int = 100
    tournament_size: int = 4
    crossover_prob: float = 0.95
    mutation_prob: float = 0.15
    mutation_type: str = "inversion"
    keep_elitism: int = 5
    seed: int = 2024
    use_greedy_seed: bool = True
    use_order_crossover: bool = True
    use_local_search: bool = True
    local_search_max_swaps: Optional[int] = 500
    crossover_type: str = "single_point"


@dataclass
class GAResult:
    best_tour: List[int]
    best_cost: float
    history: List[float]


def nearest_neighbor_tour(
    matrix: np.ndarray, start_idx: int, available_mask: np.ndarray
) -> List[int]:
    """Generates a greedy tour using open nodes."""
    unvisited = np.flatnonzero(available_mask).tolist()
    if start_idx in unvisited:
        unvisited.remove(start_idx)

    tour: List[int] = []
    current = start_idx
    while unvisited:
        distances = matrix[current, unvisited]
        next_pos = int(np.argmin(distances))
        next_node = unvisited.pop(next_pos)
        tour.append(next_node)
        current = next_node

    return tour


def build_initial_population(
    gene_space: Sequence[int],
    population_size: int,
    rng: np.random.Generator,
    seed_tour: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Builds population by mixing the greedy seed, structural variations
    of the greedy seed (to prevent elite domination stalling), and random tours.
    """
    population: List[List[int]] = []

    if seed_tour is not None and len(seed_tour) == len(gene_space):
        base_seed = [int(g) for g in seed_tour]
        population.append(base_seed)

        # Fill half the population with 2-opt style mutations of your seed
        # This gives GA high-quality building blocks to combine during crossover
        while len(population) < population_size // 2:
            mutant = list(base_seed)
            if len(mutant) > 2:
                i, j = sorted(rng.choice(len(mutant), size=2, replace=False))
                mutant[i : j + 1] = reversed(mutant[i : j + 1])
            population.append(mutant)

    # Fill the rest with uniform random permutations for global exploration
    while len(population) < population_size:
        population.append(rng.permutation(gene_space).tolist())

    return np.asarray(population, dtype=int)


def order_crossover(parents, offspring_size, ga_instance):
    # Order crossover keeps the child as a valid permutation: copy a slice from parent A,
    # then fill remaining slots with parent B in order, skipping duplicates.
    offspring = np.empty(offspring_size, dtype=parents.dtype)
    num_genes = offspring_size[1]
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]

        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]

        cut1, cut2 = sorted(
            np.random.choice(np.arange(num_genes), size=2, replace=False)
        )
        child = np.full(num_genes, -1, dtype=parents.dtype)
        child[cut1 : cut2 + 1] = parent1[cut1 : cut2 + 1]

        segment_set = set(child[cut1 : cut2 + 1].tolist())
        fill_genes = [g for g in parent2 if g not in segment_set]

        fill_idx = 0
        for i in range(num_genes):
            if child[i] == -1:
                child[i] = fill_genes[fill_idx]
                fill_idx += 1

        offspring[k, :] = child

    return offspring


def edge_recombination_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size, dtype=parents.dtype)
    num_genes = offspring_size[1]

    for k in range(offspring_size[0]):
        parent1 = parents[k % parents.shape[0]]
        parent2 = parents[(k + 1) % parents.shape[0]]

        # 1. Build Adjacency List for all edges in both parents
        adj = {i: set() for i in parent1}
        for p in (parent1, parent2):
            for i in range(num_genes):
                # Add neighbors (handling wrap-around safely)
                adj[p[i]].add(p[(i - 1) % num_genes])
                adj[p[i]].add(p[(i + 1) % num_genes])

        # 2. Build the Child Route
        child = []
        # Start with a random choice between the two parents' first nodes
        current = np.random.choice([parent1[0], parent2[0]])

        while len(child) < num_genes:
            child.append(current)

            # Remove current node from all remaining adjacency lists
            for neighbors in adj.values():
                neighbors.discard(current)

            neighbors = adj[current]

            if neighbors:
                # Choose the neighbor with the fewest remaining links (isolate mitigation)
                min_len = min(len(adj[n]) for n in neighbors)
                candidates = [n for n in neighbors if len(adj[n]) == min_len]
                current = np.random.choice(candidates)
            else:
                # If we hit a dead end, pick a random unvisited node
                remaining = [n for n in parent1 if n not in child]
                if remaining:
                    current = np.random.choice(remaining)

        offspring[k, :] = child

    return offspring


def two_opt_permutation(env: DynamicRoadTSP, tour: List[int], max_swaps: int = 300):
    """
    Asymmetric-safe 2-opt. Re-calculates full cost to properly handle
    one-ways streets and depot connections. Includes index -1 to allow
    realigning the first edge coming out of the depot.
    """
    best = tour[:]
    best_cost = env.route_cost(best)
    swaps = 0
    improved = True
    n = len(best)

    while improved:
        improved = False
        for i in range(-1, n - 1):
            for k in range(i + 2, n):
                if max_swaps is not None and swaps >= max_swaps:
                    return best, best_cost

                new_tour = best[: i + 1] + best[i + 1 : k + 1][::-1] + best[k + 1 :]
                new_cost = env.route_cost(new_tour)

                if new_cost < best_cost - 1e-4:
                    best = new_tour
                    best_cost = new_cost
                    swaps += 1
                    improved = True
                    break
            if improved:
                break

    return best, best_cost


def run_ga_pygad(env: DynamicRoadTSP, config: PygadConfig) -> GAResult:
    gene_space = env.available_indices()
    num_genes = len(gene_space)
    history: List[float] = []
    rng = np.random.default_rng(config.seed)

    def fitness_func(ga_instance: pygad.GA, solution: Sequence[int], solution_idx: int):
        tour = [int(g) for g in solution]
        cost = env.route_cost(tour)

        # Add a massive penalty if the GA omits genes or provides illegal spaces
        if len(set(tour)) != num_genes:
            return -1e10

        return -float(cost)

    def on_generation(ga_instance: pygad.GA):
        iteration = ga_instance.generations_completed

        env.update(iteration)

        best_solution, best_fitness, _ = ga_instance.best_solution()
        cost = env.route_cost([int(g) for g in best_solution])
        history.append(cost)

        # Adaptive Local Search Settings
        if iteration < 150:
            local_search_interval = 10
        elif iteration < 500:
            local_search_interval = 25
        else:
            local_search_interval = 50

        local_search_top_k = 4

        if iteration % local_search_interval == 0:
            fitness = ga_instance.last_generation_fitness.copy()
            top_idx = np.argsort(fitness)[-local_search_top_k:]

            for idx in top_idx:
                current_ind = ga_instance.population[idx].astype(int).tolist()
                improved, new_cost = two_opt_permutation(
                    env, current_ind, max_swaps=150
                )

                ga_instance.population[idx, :] = improved
                ga_instance.last_generation_fitness[idx] = -float(new_cost)

        if iteration % 50 == 0 or iteration == 1:
            logger.info(
                f"Generation {iteration}: Current Best Route Cost = {cost:.2f} seconds"
            )

    initial_population = None
    if config.use_greedy_seed:
        greedy_tour = nearest_neighbor_tour(
            env.current_matrix, env.data.start_index, env.available_mask
        )
        initial_population = build_initial_population(
            gene_space, config.population_size, rng, greedy_tour
        )

    ga = pygad.GA(
        num_generations=config.iterations,
        sol_per_pop=config.population_size,
        num_parents_mating=config.num_parents_mating,
        num_genes=num_genes,
        fitness_func=fitness_func,
        on_generation=on_generation,
        # initial_population=initial_population,
        gene_space=gene_space,
        gene_type=int,
        allow_duplicate_genes=False,
        parent_selection_type="tournament",
        K_tournament=config.tournament_size,
        crossover_type=edge_recombination_crossover
        if config.use_order_crossover
        else config.crossover_type,
        crossover_probability=config.crossover_prob,
        mutation_type=config.mutation_type,
        mutation_probability=config.mutation_prob,
        keep_elitism=config.keep_elitism,
        random_seed=config.seed,
    )

    ga.run()

    best_solution, _, _ = ga.best_solution()
    best_tour = [int(g) for g in best_solution]
    best_cost = env.route_cost(best_tour)

    if config.use_local_search:
        best_tour, best_cost = two_opt_permutation(
            env, best_tour, max_swaps=config.local_search_max_swaps
        )

    return GAResult(
        best_tour=best_tour,
        best_cost=best_cost,
        history=history,
    )


def main() -> None:
    config = ExperimentConfig()
    data, G, env = setup_experiment(config)

    ga_config = PygadConfig(
        population_size=200,
        iterations=100,
        num_parents_mating=140,
        tournament_size=5,
        crossover_prob=0.90,
        mutation_prob=0.35,
        mutation_type="scramble",
        keep_elitism=3,
        use_order_crossover=False,
        use_local_search=True,
        local_search_max_swaps=800,
        use_greedy_seed=False,
    )

    result = run_ga_pygad(env, ga_config)
    print(f"\nFinal Optimized Cost: {result.best_cost:.2f} seconds")

    plt.figure(figsize=(8, 4.5))
    plt.plot(result.history, color="royalblue", lw=2, label="Best Tour Cost")
    plt.xlabel("Generation Index")
    plt.ylabel("Cost Value (Seconds)")
    plt.title("Genetic Algorithm Cost Convergence Curve Map")
    plt.grid(True, alpha=0.4, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.show()

    osm_nodes = tour_to_route_nodes(
        G, data.node_ids, data.start_index, result.best_tour
    )
    ox.plot_graph_route(G, osm_nodes, route_color="red")


if __name__ == "__main__":
    main()
