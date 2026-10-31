"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE_SECONDS = 145.0
_start_time = time.time()

def _remaining():
    return _DEADLINE_SECONDS - (time.time() - _start_time)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city in a TSP tour construction.
    
    Strategy: 
    1. Immediate cost: d(current, candidate)
    2. 1-step Lookahead: min over next nodes n of (d(candidate, n) + d(n, destination))
    3. Geometric Detour: d(current, candidate) + d(candidate, destination) - d(current, destination)
       This penalizes candidates that deviate from the straight line between current and destination.
    
    Score = d(cur, c) + 0.5 * Lookahead(c) + 0.5 * Detour(c)
    """
    if len(unvisited_nodes) == 0:
        return int(destination_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline
    if _remaining() < 1.0:
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    # Distances from current to all unvisited
    d_cur_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # Distance from current to destination (for detour calculation)
    d_cur_to_dest = distance_matrix[current_node, destination_node]
    
    # Distances from all unvisited to destination
    d_unvisited_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Distance matrix among unvisited nodes
    # This is a view or copy, size N x N
    d_unvisited_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Calculate Lookahead costs for each candidate
    # Lookahead(c) = min_{n in unvisited, n != c} (d(c, n) + d(n, dest))
    lookahead_costs = np.zeros(n_unvisited)
    
    # If N is small, we can do this in a loop. For N=50, this is 50 * 50 = 2500 ops.
    for i in range(n_unvisited):
        # d from candidate i to all others
        d_i_to_others = d_unvisited_matrix[i, :].copy()
        d_i_to_others[i] = np.inf # Exclude self
        
        # Cost to go to n, then to dest
        # d_i_to_others + d_unvisited_to_dest
        costs = d_i_to_others + d_unvisited_to_dest
        lookahead_costs[i] = np.min(costs)
    
    # Calculate Detour costs
    # Detour(c) = d(cur, c) + d(c, dest) - d(cur, dest)
    detour_costs = d_cur_to_unvisited + d_unvisited_to_dest - d_cur_to_dest
    
    # Combine scores
    # Weights: Immediate (1.0), Lookahead (0.5), Detour (0.5)
    # The detour term is essentially a geometric alignment penalty.
    scores = d_cur_to_unvisited + 0.5 * lookahead_costs + 0.5 * detour_costs
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
