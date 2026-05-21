from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numba as nb
import numpy as np
import osmnx as ox


@dataclass(frozen=True)
class ExperimentConfig:
    place: str = "Centro, Madrid, Spain"
    # place: str = "Madrid, Spain"
    cache_dir: Path = Path("cache")
    node_count: int = 80
    sample_seed: int = 42
    start_index: int = 0
    update_interval: int = 200
    # noise_range: Tuple[float, float] = (0.9, 1.1)
    noise_range: Tuple[float, float] = (1.0, 1.0)
    drop_fraction: float = 0.05
    dynamics_seed: int = 123
    use_cache: bool = True
    log_console: bool = True


@dataclass
class ExperimentData:
    node_ids: List[int]
    base_matrix: np.ndarray
    start_index: int
    id_to_index: Dict[int, int]


@dataclass
class ExperimentState:
    iteration: int
    available_mask: np.ndarray
    current_matrix: np.ndarray


def build_drive_graph(
    place: str,
    graph_path: Path,
    use_cache: bool = True,
    log_console: bool = True,
) -> nx.MultiDiGraph:
    ox.settings.use_cache = use_cache
    ox.settings.log_console = log_console

    if graph_path.exists():
        G = ox.load_graphml(graph_path)
        edge_data = next(iter(G.edges(data=True)), None)
        if edge_data is not None and "travel_time" not in edge_data[2]:
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)

        G = ox.truncate.largest_component(G, strongly=True)
        # G = ox.project_graph(G)
        # G = ox.consolidate_intersections(G, tolerance=10, rebuild_graph=True)
        return G

    G = ox.graph_from_place(place, network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    G = ox.truncate.largest_component(G, strongly=True)
    ox.save_graphml(G, graph_path)

    return G


def sample_nodes(G: nx.MultiDiGraph, node_count: int, seed: int) -> List[int]:
    if node_count < 0:
        return list(G.nodes)

    rng = random.Random(seed)
    nodes = list(G.nodes)

    print(
        f"Sampling {node_count} nodes from {len(nodes)} total nodes with seed {seed}..."
    )

    return rng.sample(nodes, node_count)


def build_time_matrix(G: nx.MultiDiGraph, nodes: List[int]) -> np.ndarray:
    n = len(nodes)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i, source in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(G, source, weight="travel_time")
        for j, target in enumerate(nodes):
            matrix[i][j] = float(lengths[target])
    return matrix


def _validate_config(config: ExperimentConfig) -> None:
    if config.node_count < 2 and config.node_count != -1:
        raise ValueError("node_count must be at least 2 or -1 for all nodes.")
    if config.update_interval <= 0:
        raise ValueError("update_interval must be positive.")
    if not 0.0 <= config.drop_fraction < 1.0:
        raise ValueError("drop_fraction must be in [0, 1).")
    low, high = config.noise_range
    if low <= 0 or high <= 0 or low > high:
        raise ValueError("noise_range must be positive and ordered (low <= high).")


def make_experiment_data(
    config: ExperimentConfig,
) -> tuple[ExperimentData, nx.MultiDiGraph]:
    _validate_config(config)
    G = build_drive_graph(
        config.place,
        config.cache_dir / (config.place),
        config.use_cache,
        config.log_console,
    )
    node_ids = sample_nodes(G, config.node_count, config.sample_seed)
    base_matrix = build_time_matrix(G, node_ids)
    if not 0 <= config.start_index < len(node_ids):
        raise ValueError("start_index out of range for sampled nodes.")
    id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
    return ExperimentData(
        node_ids=node_ids,
        base_matrix=base_matrix,
        start_index=config.start_index,
        id_to_index=id_to_index,
    ), G


class DynamicRoadTSP:
    def __init__(self, data: ExperimentData, config: ExperimentConfig) -> None:
        self.data = data
        self.config = config
        self.current_matrix = data.base_matrix.copy()
        self.available_mask = np.ones(len(data.base_matrix), dtype=bool)
        self.available_mask[data.start_index] = True
        self._rng = np.random.default_rng(config.dynamics_seed)

    @property
    def n(self) -> int:
        return len(self.data.base_matrix)

    def update(self, iteration: int) -> bool:
        if iteration % self.config.update_interval != 0:
            return False
        self._apply_noise()
        self._update_availability()
        return True

    def _apply_noise(self) -> None:
        # To simulate dynamic conditions - dynamic travel times
        low, high = self.config.noise_range
        noise = self._rng.uniform(low, high, size=self.data.base_matrix.shape)
        self.current_matrix = self.data.base_matrix * noise
        np.fill_diagonal(self.current_matrix, 0.0)

    def _update_availability(self) -> None:
        self.available_mask = np.ones(self.n, dtype=bool)
        self.available_mask[self.data.start_index] = True
        candidates = np.array(
            [i for i in range(self.n) if i != self.data.start_index], dtype=int
        )
        drop_count = int(candidates.size * self.config.drop_fraction)
        if drop_count <= 0:
            return
        drop_indices = self._rng.choice(candidates, size=drop_count, replace=False)
        self.available_mask[drop_indices] = False

    def available_indices(self) -> List[int]:
        mask = self.available_mask.copy()
        mask[self.data.start_index] = False
        return np.flatnonzero(mask).tolist()

    def route_cost(self, tour: List[int], return_to_start: bool = True) -> float:
        return route_cost_numba(
            np.array(tour, dtype=int),
            self.current_matrix,
            self.available_mask,
            self.data.start_index,
            return_to_start,
        )

    def state(self, iteration: int) -> ExperimentState:
        self.update(iteration)
        return ExperimentState(
            iteration=iteration,
            available_mask=self.available_mask.copy(),
            current_matrix=self.current_matrix.copy(),
        )


def setup_experiment(
    config: ExperimentConfig = ExperimentConfig(),
) -> Tuple[ExperimentData, nx.MultiGraph, DynamicRoadTSP]:
    data, G = make_experiment_data(config)
    env = DynamicRoadTSP(data, config)
    return data, G, env


@nb.njit(cache=True, fastmath=True)
def route_cost_numba(
    tour: np.ndarray,
    matrix: np.ndarray,
    available: np.ndarray,
    start: int,
    return_to_start: bool = True,
) -> float:
    total = 0.0
    prev = start
    found = False

    for i in range(tour.shape[0]):
        node = tour[i]

        if node == start or not available[node]:
            continue

        total += matrix[prev, node]
        prev = node
        found = True

    if found and return_to_start:
        total += matrix[prev, start]

    return total
