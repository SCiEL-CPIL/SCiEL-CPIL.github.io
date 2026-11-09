# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 120.0
_SOFT_DEADLINE = 118.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Lookahead-2 constructive heuristic:
    Selects the next city 'u' that minimizes the sum of the cost to reach 'u' 
    and the cost of the cheapest subsequent edge from 'u' to any other unvisited city.
    
    This avoids local minima where the nearest neighbor leads to a long next edge.
    """
    # Deadline guard
    if time.time() > _SOFT_DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return current_node
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If only 2 nodes left, the choice determines the closing edge directly.
    # We should pick the one that minimizes the total remaining path to destination.
    # Path: current -> u1 -> u2 -> destination
    # Cost: d(c, u1) + d(u1, u2) + d(u2, dest)
    # Since u2 is the other node, we compare:
    # Option A: pick u1 first. Cost = d(c, u1) + d(u1, u2) + d(u2, dest)
    # Option B: pick u2 first. Cost = d(c, u2) + d(u2, u1) + d(u1, dest)
    # Note d(u1, u2) == d(u2, u1). So we compare d(c, u1) + d(u2, dest) vs d(c, u2) + d(u1, dest)
    if n_unvisited == 2:
        u1, u2 = unvisited_nodes
        cost_pick_u1 = distance_matrix[current_node, u1] + distance_matrix[u2, destination_node]
        cost_pick_u2 = distance_matrix[current_node, u2] + distance_matrix[u1, destination_node]
        if cost_pick_u1 <= cost_pick_u2:
            return int(u1)
        else:
            return int(u2)
    
    # General case: Lookahead-2
    # For each candidate u in unvisited:
    # 1. Cost to reach u: d(current, u)
    # 2. Cheapest next step from u to any v in unvisited (v != u): min_{v in unvisited, v != u} d(u, v)
    # Total score for u: d(current, u) + min_next_cost(u)
    
    # Vectorized implementation for speed
    # d_current: shape (n_unvisited,)
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    # d_matrix_subset: shape (n_unvisited, n_unvisited)
    d_subset = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # We need min of each row, but excluding the diagonal (distance to self is 0, which is invalid)
    # Set diagonal to inf
    np.fill_diagonal(d_subset, np.inf)
    
    # min_next_costs: shape (n_unvisited,)
    min_next_costs = np.min(d_subset, axis=1)
    
    # Total scores
    scores = d_current + min_next_costs
    
    idx = np.argmin(scores)
    
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
