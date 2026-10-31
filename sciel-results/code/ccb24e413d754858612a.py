"""TSP constructive heuristic: Nearest Neighbor with 2-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 112.0  # Safety margin before 120s limit

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a Nearest Neighbor strategy with a 2-step lookahead.
    For each candidate node u in unvisited_nodes, we estimate the cost as:
    dist(current, u) + min(dist(u, v) for v in unvisited_nodes if v != u)
    This avoids getting stuck in local minima of simple NN.
    """
    current_node = int(current_node)
    destination_node = int(destination_node)
    n_unvisited = len(unvisited_nodes)

    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, destination_node]
        return c1 if cost1 < cost2 else c2

    if _is_deadline_passed():
        # Fallback to simple NN if time is critical
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    # Extract distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get the submatrix of distances between unvisited nodes
    # unvisited_nodes is an array of indices
    sub_dm = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each node i in unvisited_nodes, find the minimum distance to any other node in unvisited_nodes
    # We need min(sub_dm[i, j]) for j != i
    # Set diagonal to inf to exclude self
    sub_dm_copy = sub_dm.copy()
    np.fill_diagonal(sub_dm_copy, np.inf)
    
    # min_dist_to_other[i] is the min distance from unvisited_nodes[i] to any other unvisited node
    min_dist_to_other = np.min(sub_dm_copy, axis=1)
    
    # Total lookahead cost for each candidate i
    # cost[i] = dist(current, unvisited[i]) + min_dist_to_other[i]
    lookahead_costs = dist_from_current + min_dist_to_other
    
    # Find the index with minimum cost
    best_idx = int(np.argmin(lookahead_costs))
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
