from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from matplotlib.animation import FuncAnimation

from utils import tour_to_route_nodes


def plot_graph(
    G: nx.MultiDiGraph,
    node_size: float = 0.5,
    edge_linewidth: float = 0.5,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = ox.plot_graph(
        G,
        node_size=node_size,
        node_color="lightgray",
        edge_color="lightgray",
        edge_linewidth=edge_linewidth,
        bgcolor="white",
        show=False,
        close=False,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_tour_on_graph(
    G: nx.MultiDiGraph,
    node_ids: Sequence[int],
    start_index: int,
    tour: Sequence[int],
    route_color: str = "red",
    route_linewidth: float = 3.0,
    node_size: float = 0.0,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    route_nodes = tour_to_route_nodes(G, node_ids, start_index, tour)
    fig, ax = ox.plot_graph_route(
        G,
        route_nodes,
        route_color=route_color,
        route_linewidth=route_linewidth,
        node_size=node_size,
        bgcolor="white",
        show=False,
        close=False,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def animate_tour_progression(
    G: nx.MultiDiGraph,
    node_ids: Sequence[int],
    start_index: int,
    tours: Sequence[Sequence[int]],
    iterations: Optional[Sequence[int]] = None,
    weight: str = "travel_time",
    interval: int = 700,
    route_color: str = "red",
    route_linewidth: float = 3.0,
    node_size: float = 0.5,
    edge_linewidth: float = 0.5,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> FuncAnimation:
    if iterations is None:
        iterations = list(range(len(tours)))

    fig, ax = ox.plot_graph(
        G,
        node_size=node_size,
        node_color="lightgray",
        edge_color="lightgray",
        edge_linewidth=edge_linewidth,
        bgcolor="white",
        show=False,
        close=False,
    )

    (route_line,) = ax.plot(
        [], [], color=route_color, linewidth=route_linewidth, zorder=3
    )

    def update(frame_idx: int):
        tour = tours[frame_idx]
        route_nodes = build_route_nodes(G, node_ids, start_index, tour, weight=weight)
        xs = [G.nodes[n]["x"] for n in route_nodes]
        ys = [G.nodes[n]["y"] for n in route_nodes]
        route_line.set_data(xs, ys)
        ax.set_title(f"Iteration {iterations[frame_idx]}")
        return (route_line,)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(tours),
        interval=interval,
        blit=True,
    )

    if save_path is not None:
        ani.save(save_path, dpi=200)

    if show:
        plt.show()

    return ani
