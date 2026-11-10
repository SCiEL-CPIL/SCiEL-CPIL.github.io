"""CVRP Heuristic: Savings-based with 2-step Lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0
_DEADLINE = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a savings heuristic with a 2-step lookahead.
    This allows the heuristic to plan one step further ahead, potentially
    finding better chains of nodes that fit within capacity constraints.
    """
    global _start_time, _DEADLINE
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes, dtype=int)
    else:
        unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)
    
    # Find feasible nodes
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    # If no feasible nodes, return depot
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])
    
    # Pre-fetch distances and demands for vectorization
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    demands_feasible = demands[feasible_nodes]
    
    # 1. Calculate Immediate Savings: S(C, J)
    # S(C, J) = d(C, Depot) + d(Depot, J) - d(C, J)
    savings_1 = d_current_depot + d_depot_j - d_current_j
    
    # 2. Calculate 2-step Lookahead
    # For each J, find the best K such that K is feasible after J.
    # We need to compute savings S(J, K) for all pairs (J, K).
    # S(J, K) = d(J, Depot) + d(Depot, K) - d(J, K)
    
    # d(J, Depot) is d_depot_j (symmetric)
    # d(Depot, K) is d_depot_j
    # d(J, K) is distance_matrix[feasible_nodes[:, None], feasible_nodes[None, :]]
    
    # Create distance matrix between all feasible nodes
    # d_jk[i, k] = distance(feasible_nodes[i], feasible_nodes[k])
    d_jk = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # Calculate savings matrix S(J, K)
    # S(J, K) = d_depot_j[:, None] + d_depot_j[None, :] - d_jk
    savings_2_matrix = d_depot_j[:, None] + d_depot_j[None, :] - d_jk
    
    # Mask invalid K for each J:
    # 1. K cannot be J (diagonal)
    # 2. Demand of K must be <= rest_capacity - Demand of J
    
    # remaining_cap_after_j = rest_capacity - demands_feasible
    # We need demands_feasible[k] <= remaining_cap_after_j[i]
    
    # Create a mask for capacity feasibility
    # cap_mask[i, k] is True if demands_feasible[k] <= (rest_capacity - demands_feasible[i])
    
    # Use broadcasting
    # demands_feasible is (N,)
    # rest_capacity is scalar
    # remaining_caps is (N,)
    remaining_caps = rest_capacity - demands_feasible
    
    # cap_mask[i, k] = demands_feasible[k] <= remaining_caps[i]
    cap_mask = demands_feasible[None, :] <= remaining_caps[:, None]
    
    # Exclude self (diagonal)
    np.fill_diagonal(cap_mask, False)
    
    # Find the max savings for each J over all valid K
    # Set invalid K to -inf
    valid_savings_2 = np.where(cap_mask, savings_2_matrix, -np.inf)
    
    # max_lookahead_2[i] is the best 2nd step saving if we pick J_i
    # If no valid K exists, this will be -inf, which we should treat as 0 or -inf depending on preference.
    # If no K exists, the route ends after J. So the "lookahead" contribution is 0 (no further savings).
    # However, if savings_1 is positive, we might still want to go to J even if we can't add K.
    # So we should clip the max to 0 if no valid K exists? Or just let it be -inf and handle later?
    # If we use -inf, and savings_1 is positive, total_score will be -inf, which is bad.
    # So, if no valid K, lookahead should be 0.
    
    max_lookahead_2 = np.max(valid_savings_2, axis=1)
    
    # If no valid K was found for a row, max will be -inf.
    # We want to treat "no further step" as 0 additional score from lookahead.
    # So we replace -inf with 0.
    max_lookahead_2 = np.where(np.isneginf(max_lookahead_2), 0, max_lookahead_2)
    
    # Combine scores
    # Weight alpha for the lookahead
    alpha = 0.5
    total_score = savings_1 + alpha * max_lookahead_2
    
    # If the best total score is negative, it's better to close the route
    if np.max(total_score) < 0:
        return int(depot)
    
    max_idx = np.argmax(total_score)
    
    return int(feasible_nodes[max_idx])
# EVOLVE-BLOCK-END
