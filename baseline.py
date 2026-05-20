from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import osmnx as ox

from data import DynamicRoadTSP, ExperimentConfig, setup_experiment
from utils import tour_to_route_nodes


@dataclass
class BaselineResult:
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


def run_nearest_neighbor_baseline(
    env: DynamicRoadTSP,
    iterations: int,
    recompute_on_update: bool = True,
) -> BaselineResult:
    current_tour = nearest_neighbor_tour(
        env.current_matrix, env.data.start_index, env.available_mask
    )
    best_tour = current_tour[:]
    best_cost = env.route_cost(current_tour)

    history: List[float] = [best_cost]

    for iteration in range(1, iterations + 1):
        updated = env.update(iteration)
        if updated and recompute_on_update:
            current_tour = nearest_neighbor_tour(
                env.current_matrix, env.data.start_index, env.available_mask
            )

        cost = env.route_cost(current_tour)
        if cost < best_cost:
            best_cost = cost
            best_tour = current_tour[:]

        history.append(cost)

    return BaselineResult(best_tour=best_tour, best_cost=best_cost, history=history)


def main() -> None:
    config = ExperimentConfig(node_count=-1)
    data, G, env = setup_experiment(config)

    n_iterations = 200
    result = run_nearest_neighbor_baseline(env, iterations=n_iterations)
    print(f"Baseline best cost: {result.best_cost:.2f} seconds")

    osm_nodes = tour_to_route_nodes(
        G, data.node_ids, data.start_index, result.best_tour
    )
    ox.plot_graph_route(G, osm_nodes, route_color="red")


if __name__ == "__main__":
    main()
