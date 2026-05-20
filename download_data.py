# import matplotlib.pyplot as plt
# import osmnx as ox

# ox.settings.use_cache = True
# ox.settings.log_console = True

# G = ox.graph_from_place("Madrid, Spain", network_type="drive")

# G = ox.add_edge_speeds(G)
# G = ox.add_edge_travel_times(G)

# # Convert to GeoDataFrames
# nodes, edges = ox.graph_to_gdfs(G)

# nodes.to_file("madrid_drive_nodes.gpkg", layer="nodes", driver="GPKG")

# fig, ax = ox.plot_graph(
#     G,
#     node_size=0.5,
#     node_color="black",
#     edge_color="gray",
#     edge_linewidth=0.5,
#     bgcolor="white",
#     show=False,
#     close=False,
# )
# fig.savefig("madrid_drive_network.png", dpi=300, bbox_inches="tight")
# plt.show()
