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
    population_size: int = 120
    iterations: int = 200
    num_parents_mating: int = 60
    tournament_size: int = 3
    crossover_prob: float = 0.9
    mutation_prob: float = 0.2
    mutation_type: str = "inversion"
    keep_elitism: int = 2
    seed: int = 2024
    use_greedy_seed: bool = False
    use_order_crossover: bool = True
    use_local_search: bool = True
    local_search_max_swaps: Optional[int] = 2000
    crossover_type: str = "single_point"


@dataclass
class GAResult:
    best_tour: List[int]
    best_cost: float
    history: List[float]


def nearest_neighbor_tour(
    matrix: np.ndarray, start_idx: int, available_mask: np.ndarray
) -> List[int]:
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
    population: List[List[int]] = []
    if seed_tour is not None and len(seed_tour) == len(gene_space):
        population.append([int(g) for g in seed_tour])
        if len(population) < population_size:
            population.append(list(reversed(population[0])))

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


def two_opt_permutation(env, tour, max_swaps=200):
    best = tour[:]
    best_cost = env.route_cost(best)
    swaps = 0
    improved = True

    while improved:
        improved = False
        n = len(best)
        for i in range(n - 1):
            for k in range(i + 2, n):
                if max_swaps is not None and swaps >= max_swaps:
                    return best, best_cost
                swaps += 1

                # Calculate cost delta for reversing best[i+1:k+1]
                # Only 4 edges are affected:
                # Remove: best[i]->best[i+1] and best[k]->best[k+1]
                # Add:    best[i]->best[k] and best[i+1]->best[k+1]
                a = best[i]
                b = best[i + 1]
                c = best[k]
                d = best[(k + 1) % n]

                old_cost = env.current_matrix[a, b] + env.current_matrix[c, d]
                new_cost = env.current_matrix[a, c] + env.current_matrix[b, d]
                delta = new_cost - old_cost

                # Accept improvement (with numerical tolerance)
                if delta < -1e-9:
                    best[i + 1 : k + 1] = reversed(best[i + 1 : k + 1])
                    best_cost += delta
                    improved = True
                    break

            if improved:
                break

    return best, best_cost


def run_ga_pygad(env: DynamicRoadTSP, config: PygadConfig) -> GAResult:
    # gene_space = [i for i in range(env.n) if i != env.data.start_index]
    gene_space = env.available_indices()
    num_genes = len(gene_space)
    history: List[float] = []
    rng = np.random.default_rng(config.seed)

    def _to_int_tour(solution: Sequence[int]) -> List[int]:
        return [int(g) for g in solution]

    def fitness_func(ga_instance: pygad.GA, solution: Sequence[int], solution_idx: int):
        cost = env.route_cost(list(solution))
        return -float(cost)

    def on_generation(ga_instance: pygad.GA):
        iteration = ga_instance.generations_completed + 1

        best_solution, best_fitness, _ = ga_instance.best_solution()
        cost = env.route_cost(best_solution)
        history.append(cost)

        # Adaptive local search interval
        if iteration < 100:
            local_search_interval = 20  # Early: aggressive
        elif iteration < 300:
            local_search_interval = 50  # Mid: moderate
        else:
            local_search_interval = 100  # Late: sparse
        local_search_top_k = 5

        if iteration % local_search_interval == 0:
            fitness = ga_instance.last_generation_fitness
            top_idx = np.argsort(fitness)[-local_search_top_k:]
            for idx in top_idx:
                improved, _ = two_opt_permutation(
                    env, ga_instance.population[idx].tolist(), max_swaps=500
                )
                ga_instance.population[idx, :] = improved

        if iteration % 100 == 0:
            logger.info(f"Gen {iteration}: Best cost = {cost:.2f} seconds")

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
        gene_space=gene_space,
        gene_type=int,
        allow_duplicate_genes=False,
        parent_selection_type="rank",
        K_tournament=config.tournament_size,
        crossover_type=order_crossover,
        # if config.use_order_crossover
        # else config.crossover_type,
        crossover_probability=config.crossover_prob,
        mutation_type=config.mutation_type,
        mutation_probability=config.mutation_prob,
        keep_elitism=config.keep_elitism,
        random_seed=config.seed,
        # parallel_processing=["thread", 8],
    )

    ga.run()

    best_solution, _, _ = ga.best_solution()
    best_tour = _to_int_tour(best_solution)
    best_cost = env.route_cost(best_tour)

    if config.use_local_search:
        improved_tour, improved_cost = two_opt_permutation(
            env, best_tour, max_swaps=config.local_search_max_swaps
        )
        if improved_cost < best_cost:
            best_tour = improved_tour
            best_cost = improved_cost

    return GAResult(
        best_tour=best_tour,
        best_cost=best_cost,
        history=history,
    )


def main() -> None:
    config = ExperimentConfig()
    data, G, env = setup_experiment(config)

    ga_config = PygadConfig(
        population_size=100,
        iterations=2000,
        num_parents_mating=60,
        tournament_size=4,
        crossover_prob=0.95,
        mutation_prob=0.1,
        keep_elitism=2,
        use_order_crossover=True,
        use_local_search=False,
        local_search_max_swaps=500,
        use_greedy_seed=True,
    )

    result = run_ga_pygad(env, ga_config)
    print(f"PyGAD best cost: {result.best_cost:.2f} seconds")

    plt.figure(figsize=(7, 4))
    plt.plot(result.history, label="Best cost per iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (seconds)")
    plt.title("GA convergence (200 iterations)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    osm_nodes = tour_to_route_nodes(
        G, data.node_ids, data.start_index, result.best_tour
    )
    ox.plot_graph_route(G, osm_nodes, route_color="red")


if __name__ == "__main__":
    main()
