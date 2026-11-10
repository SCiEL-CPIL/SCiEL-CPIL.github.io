"""Combined CVRP heuristic: Savings-based selection with 1-step Lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a savings heuristic with a 1-step lookahead.
    The lookahead estimates the best potential next saving from the candidate node,
    encouraging the selection of nodes that maintain a chain of high-savings connections.
    """
    global _start_time, _DEADLINE
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array if it's a list
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes, dtype=int)
    else:
        unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)
    
    # Find feasible nodes: those whose demand fits in rest_capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    # If no feasible nodes, return depot
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])
    
    n_feasible = len(feasible_nodes)
    
    # Compute distances
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    
    # Calculate immediate savings for each feasible node
    savings = d_current_depot + d_depot_j - d_current_j
    
    # Lookahead: For each candidate j, estimate the best saving from j to any other feasible node k
    # This is O(N^2) but N is small (<=50).
    # We only consider k that are feasible given the remaining capacity after visiting j.
    # rest_capacity_after_j = rest_capacity - demands[j]
    
    lookahead_scores = np.zeros(n_feasible)
    
    # Pre-fetch demand for feasible nodes
    demands_feasible = demands[feasible_nodes]
    
    for i, j in enumerate(feasible_nodes):
        remaining_cap = rest_capacity - demands_feasible[i]
        
        # If no capacity left for others, lookahead is 0
        if remaining_cap <= 0:
            continue
            
        # Find nodes k (in feasible_nodes) that are not j and fit in remaining_cap
        # Note: We use the same feasible_nodes list, so we need to mask out j and check capacity
        # Since demands are positive, if k fits in remaining_cap, it's a valid next step.
        
        # Vectorized calculation for lookahead
        # d_j_k = distance_matrix[j, feasible_nodes]
        # d_k_depot = d_depot_j
        # d_j_depot = d_depot_j[i]
        
        d_j_all = distance_matrix[j, feasible_nodes]
        
        # Calculate savings from j to all k in feasible_nodes
        # Savings(j, k) = d(j, depot) + d(depot, k) - d(j, k)
        savings_j_k = d_depot_j[i] + d_depot_j - d_j_all
        
        # Mask: k cannot be j, and demands[k] must be <= remaining_cap
        mask_k = np.ones(n_feasible, dtype=bool)
        mask_k[i] = False # Exclude j itself
        mask_k &= (demands_feasible <= remaining_cap)
        
        if np.any(mask_k):
            # Get max savings among valid k
            # Set invalid to -inf
            valid_savings = np.where(mask_k, savings_j_k, -np.inf)
            max_lookahead = np.max(valid_savings)
            # Only add positive lookahead
            if max_lookahead > 0:
                lookahead_scores[i] = max_lookahead
        else:
            lookahead_scores[i] = 0
    
    # Combine immediate savings and lookahead
    # Weight for lookahead: 0.5 seems reasonable to not be too greedy
    alpha = 0.5
    total_score = savings + alpha * lookahead_scores
    
    # If the best total score is negative, it's better to close the route
    # Note: If savings is negative but lookahead is high, it might still be worth it?
    # Usually, if immediate savings is very negative, the route is inefficient.
    # Let's stick to: if max(total_score) < 0, close.
    
    if np.max(total_score) < 0:
        return int(depot)
    
    max_idx = np.argmax(total_score)
    
    return int(feasible_nodes[max_idx])
# EVOLVE-BLOCK-END
