"""Direct heuristic for TSP next-node selection."""
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s hard kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Structural change: Pure Nearest Neighbor for most of the tour,
    switching to destination-biased selection near the end to close the loop efficiently.
    """
    # Time guard
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        u, v = unvisited_nodes
        # Cost of closing via u then v, or v then u
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
    
    # Threshold for switching to destination-aware mode
    # Knowledge suggests blending in the last ~10-15 picks
    if n_unvisited > 15:
        # Pure Nearest Neighbor
        d_curr = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(d_curr)
        return int(unvisited_nodes[best_idx])
    else:
        # Blend current distance and destination distance
        # Weight on destination increases as n_unvisited decreases
        # At n=15, alpha=0.1; at n=3, alpha=0.7
        alpha = 0.1 + 0.6 * (1.0 - (n_unvisited - 1) / 14.0)
        
        d_curr = distance_matrix[current_node, unvisited_nodes]
        d_dest = distance_matrix[unvisited_nodes, destination_node]
        
        scores = (1.0 - alpha) * d_curr + alpha * d_dest
        best_idx = np.argmin(scores)
        return int(unvisited_nodes[best_idx])
