"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a vectorized 3-step lookahead heuristic.
    """
    if _time_remaining() < 0.5:
        if len(unvisited_nodes) == 0:
            return depot
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        dists = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(dists)])

    if len(unvisited_nodes) == 0:
        return depot

    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if len(feasible) == 0:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible)
    n_feasible = len(feasible_arr)
    
    d_cur_to_f = distance_matrix[current_node, feasible_arr]
    d_f_to_f = distance_matrix[np.ix_(feasible_arr, feasible_arr)]
    d_f_to_depot = distance_matrix[feasible_arr, depot]
    demands_feasible = demands[feasible_arr]
    
    # Precompute M[j, k] = d(j, k) + d(k, depot)
    # This represents the cost of going from j to k and then back to depot.
    M = d_f_to_f + d_f_to_depot[None, :]
    
    scores = np.zeros(n_feasible)
    
    for i_idx in range(n_feasible):
        node_i = feasible_arr[i_idx]
        cost_to_i = d_cur_to_f[i_idx]
        
        remaining_cap_after_i = rest_capacity - demands[node_i]
        
        # Mask for valid j: demands[j] <= remaining_cap_after_i
        valid_j_mask = demands_feasible <= remaining_cap_after_i
        valid_j_mask[i_idx] = False # Exclude i
        
        if not np.any(valid_j_mask):
            # No valid j, so we must go back to depot
            scores[i_idx] = cost_to_i + d_f_to_depot[i_idx]
            continue
        
        # For each valid j, we need the best k.
        # remaining_cap_after_j = remaining_cap_after_i - demands[j]
        # valid_k_mask_j: demands[k] <= remaining_cap_after_j
        
        # Let's vectorize over j.
        # Create a 3D mask? Or loop over j (N=50 is small).
        
        best_j_cost = d_f_to_depot[i_idx] # Default: go back to depot from i
        
        # Loop over valid j
        valid_j_indices = np.where(valid_j_mask)[0]
        
        for j_idx in valid_j_indices:
            node_j = feasible_arr[j_idx]
            cost_i_to_j = d_f_to_f[i_idx, j_idx]
            
            remaining_cap_after_j = remaining_cap_after_i - demands[node_j]
            
            # Find best k for this j
            valid_k_mask = demands_feasible <= remaining_cap_after_j
            valid_k_mask[j_idx] = False # Exclude j
            valid_k_mask[i_idx] = False # Exclude i (already visited)
            
            if np.any(valid_k_mask):
                # Get min M[j_idx, k] for valid k
                masked_M_row = np.where(valid_k_mask, M[j_idx, :], np.inf)
                best_k_cost = np.min(masked_M_row)
            else:
                best_k_cost = d_f_to_depot[j_idx] # Go back to depot from j
            
            total_j_cost = cost_i_to_j + best_k_cost
            
            if total_j_cost < best_j_cost:
                best_j_cost = total_j_cost
        
        scores[i_idx] = cost_to_i + best_j_cost
    
    min_idx = np.argmin(scores)
    return int(feasible_arr[min_idx])

# EVOLVE-BLOCK-END
