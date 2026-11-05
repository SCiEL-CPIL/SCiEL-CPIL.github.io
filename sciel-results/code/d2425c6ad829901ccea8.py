"""Improved constructive TSP heuristic: return-aware nearest neighbour."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    dm = distance_matrix
    # Forward cost: distance from current to each candidate
    d_forward = dm[current_node, unv]
    # Return cost: distance from each candidate to destination
    d_return = dm[unv, destination_node]
    
    # Blend: prefer nodes that are close to current AND close to destination
    # Weight of 0.5 balances forward progress with tour closure
    score = d_forward + 0.5 * d_return
    
    return int(unv[np.argmin(score)])
# EVOLVE-BLOCK-END
