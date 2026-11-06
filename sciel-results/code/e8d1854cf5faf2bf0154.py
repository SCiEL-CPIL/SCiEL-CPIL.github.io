"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if _time_exceeded():
        # Fallback to nearest neighbor if time is running out
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # 1. Calculate immediate distance from current to all unvisited
    d_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Calculate distance from all unvisited to destination
    d_to_destination = distance_matrix[unvisited_nodes, destination_node]
    
    # 3. Pre-filter: Use a simple score to select top K candidates
    # We use a weighted sum to balance immediate and exit costs
    # alpha=1.0 is a strong lookahead
    alpha = 1.0
    pre_scores = d_from_current + alpha * d_to_destination
    
    # Select top K candidates
    K = min(10, n_unvisited)
    if K == n_unvisited:
        top_indices = np.arange(n_unvisited)
    else:
        # Get indices of the K smallest pre_scores
        top_indices = np.argpartition(pre_scores, K - 1)[:K]
    
    # 4. For each top candidate, perform a 2-step lookahead
    # Cost = d(current, c) + min_{n in unvisited \ {c}} (d(c, n) + d(n, destination))
    best_candidate = -1
    best_cost = float('inf')
    
    for idx in top_indices:
        c = unvisited_nodes[idx]
        
        # If this is the last node, the cost is just d(current, c) + d(c, dest)
        if n_unvisited == 1:
            cost = d_from_current[idx] + d_to_destination[idx]
        else:
            # Get remaining unvisited nodes excluding c
            # We need to find min_{n in U_rem} (d(c, n) + d(n, destination))
            # d(c, n) is distance_matrix[c, unvisited_nodes]
            # d(n, destination) is distance_matrix[unvisited_nodes, destination_node]
            
            # Vectorized calculation for all n in unvisited_nodes
            # We must exclude n == c
            d_c_to_unvisited = distance_matrix[c, unvisited_nodes]
            
            # Mask out the candidate c itself
            # The index of c in unvisited_nodes is idx
            # We want to minimize d_c_to_unvisited[j] + d_to_destination[j] for j != idx
            
            # Create a mask
            mask = np.ones(n_unvisited, dtype=bool)
            mask[idx] = False
            
            # Calculate scores for remaining nodes
            # We can just take the min of the sum, ignoring the masked out element by setting it to inf
            scores = d_c_to_unvisited + d_to_destination
            scores[idx] = float('inf')
            
            min_next_cost = np.min(scores)
            
            # Total cost for this candidate c
            cost = d_from_current[idx] + min_next_cost
            
        if cost < best_cost:
            best_cost = cost
            best_candidate = c
            
    if best_candidate == -1:
        # Fallback
        return int(unvisited_nodes[np.argmin(pre_scores)])
        
    return int(best_candidate)
# EVOLVE-BLOCK-END
