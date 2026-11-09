"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for distance matrix
_dm_cache = {}
_start_time = None
_instance_id = 0

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix)
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > 140

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: Multi-criteria lookahead combining:
    1. Immediate distance from current to candidate
    2. Continuation cost (min distance to other feasible or depot)
    3. Savings-based merge potential
    4. Demand-weighted priority (visit high-demand nodes earlier when close)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard
    if _check_deadline():
        feasible = [n for n in unvisited if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        feasible_arr = np.array(feasible)
        dists = dm[current_node, feasible_arr]
        best_idx = np.argmin(dists)
        return int(feasible_arr[best_idx])
    
    # Filter to capacity-feasible nodes
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible)
    n = len(feasible_arr)
    
    # Compute distances from current node to all feasible nodes
    dists_to_current = dm[current_node, feasible_arr]
    
    # Compute distances from each feasible node to depot
    dists_to_depot = dm[feasible_arr, depot]
    
    # Compute sub-matrix of distances among feasible nodes
    sub_dm = dm[np.ix_(feasible_arr, feasible_arr)]
    
    # Continuation cost: min distance from candidate to any other feasible node or depot
    sub_dm_no_self = sub_dm.copy()
    np.fill_diagonal(sub_dm_no_self, np.inf)
    min_to_other = np.min(sub_dm_no_self, axis=1)
    continuation = np.minimum(min_to_other, dists_to_depot)
    
    # Savings-based lookahead
    d_depot = dists_to_depot
    savings_matrix = d_depot[:, None] + d_depot[None, :] - sub_dm
    np.fill_diagonal(savings_matrix, -np.inf)
    max_savings = np.max(savings_matrix, axis=1)
    
    # Normalize savings
    max_s = np.max(np.abs(max_savings)) if np.max(np.abs(max_savings)) > 0 else 1.0
    norm_savings = max_savings / max_s
    
    # Demand-weighted priority: prefer higher demand nodes that are close
    # This encourages packing high-demand items early in routes
    demands_arr = np.array([demands[i] for i in feasible_arr], dtype=float)
    max_demand = np.max(demands_arr) if np.max(demands_arr) > 0 else 1.0
    norm_demand = demands_arr / max_demand
    
    # Isolation penalty: average distance to other feasible nodes
    if n > 1:
        sub_dm_zero_diag = sub_dm.copy()
        np.fill_diagonal(sub_dm_zero_diag, 0)
        avg_dist_to_others = np.sum(sub_dm_zero_diag, axis=1) / (n - 1)
    else:
        avg_dist_to_others = np.array([0.0])
    
    # Normalize continuation and isolation
    max_cont = np.max(continuation) if np.max(continuation) > 0 else 1.0
    norm_continuation = continuation / max_cont
    
    max_iso = np.max(avg_dist_to_others) if np.max(avg_dist_to_others) > 0 else 1.0
    norm_isolation = avg_dist_to_others / max_iso
    
    # Composite score:
    # - dist_to_current: immediate travel cost
    # - alpha * continuation: future extension cost estimate
    # - gamma * isolation: penalty for isolated nodes
    # - delta * norm_savings: reward for merge potential
    # - beta * norm_demand * dist_to_current: slight preference for high-demand close nodes
    alpha = 0.4
    gamma = 0.3
    delta = 0.35
    beta = 0.15
    
    scores = (dists_to_current 
              + alpha * norm_continuation * max_cont 
              + gamma * norm_isolation * max_iso 
              - delta * norm_savings * max_s 
              + beta * norm_demand * dists_to_current)
    
    # Pick the feasible node with the lowest score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
