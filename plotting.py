from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from matplotlib.animation import FuncAnimation


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


def build_route_nodes(
    G: nx.MultiDiGraph,
    node_ids: Sequence[int],
    start_index: int,
    tour: Sequence[int],
    weight: str = "travel_time",
) -> List[int]:
    ordered_indices = (
        [start_index] + [i for i in tour if i != start_index] + [start_index]
    )
    osm_nodes = [node_ids[i] for i in ordered_indices]

    full_route: List[int] = []
    for u, v in zip(osm_nodes, osm_nodes[1:]):
        segment = ox.shortest_path(G, u, v, weight=weight)
        if segment is None:
            raise ValueError(f"No path between {u} and {v}.")
        if full_route:
            full_route.extend(segment[1:])
        else:
            full_route.extend(segment)

    return full_route


def plot_tour_on_graph(
    G: nx.MultiDiGraph,
    node_ids: Sequence[int],
    start_index: int,
    tour: Sequence[int],
    weight: str = "travel_time",
    route_color: str = "red",
    route_linewidth: float = 3.0,
    node_size: float = 0.0,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    route_nodes = build_route_nodes(G, node_ids, start_index, tour, weight=weight)
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


def plot_graph_route_with_arrows(G, route_nodes, arrow_step=20):
    fig, ax = ox.plot_graph_route(
        G,
        route_nodes,
        route_color="red",
        route_linewidth=3,
        node_size=0,
        bgcolor="white",
        show=False,
        close=False,
    )

    xs = [G.nodes[n]["x"] for n in route_nodes]
    ys = [G.nodes[n]["y"] for n in route_nodes]

    for i in range(0, len(xs) - 1, arrow_step):
        ax.annotate(
            "",
            xy=(xs[i + 1], ys[i + 1]),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            zorder=4,
        )

    plt.show()
    return fig, ax
