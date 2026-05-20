import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import osmnx as ox


def tour_to_route_nodes(G, node_ids, start_index, tour):
    # The solver's "tour" is just an order of visit indices (or node IDs), not a real
    # street-by-street path. Consecutive nodes in that tour are usually NOT connected
    # by an edge in the graph. `tour_to_route_nodes` expands each hop into the actual
    # shortest-path sequence of graph nodes, producing a continuous, edge-by-edge
    # route that `ox.plot_graph_route` can draw without errors.
    ordered = [start_index] + [i for i in tour if i != start_index] + [start_index]
    osm_nodes = [node_ids[i] for i in ordered]

    route_nodes = []
    for u, v in zip(osm_nodes, osm_nodes[1:]):
        segment = ox.shortest_path(G, u, v, weight="travel_time")
        route_nodes.extend(segment[1:] if route_nodes else segment)

    return route_nodes


def tour_indices_to_node_ids(node_ids, start_index, tour, return_to_start=True):
    """
    Convert a tour of indices into actual OSM node IDs.

    - node_ids: list of OSM node IDs (same order as your matrix)
    - tour: list of indices (0..n-1)
    """
    ordered = [start_index] + [i for i in tour if i != start_index] + start_index
    if return_to_start:
        ordered.append(start_index)
    return [node_ids[i] for i in ordered]


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def append_run_result(
    result: Any,
    out_path: str | Path = "results/results.jsonl",
    run_name: str | None = None,
    config: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Append a single run's result to a JSONL file.

    Each call writes one JSON object per line:
      {"timestamp": "...", "run_name": "...", "config": {...}, "result": {...}}
    """
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "result": _json_safe(result),
    }
    if run_name is not None:
        payload["run_name"] = run_name
    if config is not None:
        payload["config"] = _json_safe(config)
    if metadata:
        payload["metadata"] = _json_safe(metadata)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return out_path
