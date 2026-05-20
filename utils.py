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
