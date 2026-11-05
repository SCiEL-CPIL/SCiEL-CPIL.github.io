"""Improved CVRP heuristic with dynamic regret-minimization framework."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 148.0  # Safety margin before 150s limit

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # If time is running out, return depot to close route safely
    if not _time_ok():
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Convert to numpy arrays for efficiency
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # Filter feasible nodes based on capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)
    
    feas_nodes = unvisited_arr[feas_mask]
    n_nodes = len(feas_nodes)
    
    if n_nodes == 1:
        return int(feas_nodes[0])
    
    # Calculate distances
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_feas = distance_matrix[depot, feas_nodes]
    d_cur_feas = distance_matrix[current_node, feas_nodes]
    
    # Calculate Savings
    # Savings = d[current, depot] + d[depot, c] - d[current, c]
    savings = d_cur_dep + d_dep_feas - d_cur_feas
    
    # Dynamic Regret-Minimization Framework with Lookahead:
    # Score combines:
    # 1. Savings (Clarke-Wright)
    # 2. Distance penalty (regret-minimization: prefer shorter edges)
    # 3. Lookahead term: penalize candidates whose next nearest neighbor is far
    
    # Lookahead: for each candidate, find the min distance to any other feasible node or depot
    # This estimates the "continuation cost" if we pick this node
    # We want to minimize: d(cur, cand) + min(d(cand, next), d(cand, depot))
    # So we penalize candidates with high continuation cost
    
    # Efficiently compute min distance from each candidate to (depot or other feasible nodes)
    # For each candidate i, continuation_cost = min(d(cand_i, depot), min_j(d(cand_i, cand_j)))
    
    # d(cand, depot) is d_dep_feas (distance from depot to candidate, symmetric)
    # min_j(d(cand_i, cand_j)) for i != j
    
    if n_nodes > 1:
        # Distance matrix among feasible nodes
        d_feas_feas = distance_matrix[np.ix_(feas_nodes, feas_nodes)]
        # For each row i, find min over j != i
        # Set diagonal to inf to exclude self
        np.fill_diagonal(d_feas_feas, np.inf)
        min_to_other = np.min(d_feas_feas, axis=1)
        
        # Continuation cost = min(d(cand, depot), min_to_other)
        cont_cost = np.minimum(d_dep_feas, min_to_other)
    else:
        cont_cost = d_dep_feas.copy()
    
    # Normalize terms to have comparable scales
    mean_savings = np.mean(savings)
    mean_dist = np.mean(d_cur_feas)
    mean_cont = np.mean(cont_cost)
    
    # Normalize each term to [0, 1] range for weighting
    savings_norm = (savings - np.min(savings)) / (np.max(savings) - np.min(savings) + 1e-10)
    dist_norm = (d_cur_feas - np.min(d_cur_feas)) / (np.max(d_cur_feas) - np.min(d_cur_feas) + 1e-10)
    cont_norm = (cont_cost - np.min(cont_cost)) / (np.max(cont_cost) - np.min(cont_cost) + 1e-10)
    
    # Weighted score: maximize savings, minimize distance, minimize continuation
    W_sav = 1.0
    W_dist = 0.8
    W_cont = 0.4
    
    scores = W_sav * savings_norm - W_dist * dist_norm - W_cont * cont_norm
    
    # Select the node with the highest score
    best_idx = np.argmax(scores)
    best_node = int(feas_nodes[best_idx])
    
    return best_node
# EVOLVE-BLOCK-END
