import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node to visit using a 2-step lookahead heuristic.
    For each candidate, it estimates the cost of visiting it and then the best
    subsequent node, aiming to minimize the total path segment cost.
    """
    if len(unvisited_nodes) == 0:
        return int(depot)
        
    unvisited_arr = np.array(unvisited_nodes)
    demands_arr = np.array(demands)
    dm = np.array(distance_matrix)
    
    node_demands = demands_arr[unvisited_arr]
    feasible_mask = node_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return int(depot)
        
    feasible_nodes = unvisited_arr[feasible_mask]
    
    best_score = float('inf')
    best_node = int(depot)
    
    for j in feasible_nodes:
        # Cost to reach j
        cost_to_j = dm[current_node, j]
        
        # Check if we can visit another node after j
        # Remaining capacity after j
        rem_cap_after_j = rest_capacity - demands_arr[j]
        
        # Find feasible nodes after j
        # We need to exclude j from unvisited for this lookahead
        # Since we are just estimating, we can check all unvisited nodes != j
        # and check capacity
        
        # If no capacity left, we must return to depot
        # Base cost to return to depot immediately
        min_future_cost = dm[j, depot]
        
        # Check if we can visit another node after j
        # Remaining capacity after j
        rem_cap_after_j = rest_capacity - demands_arr[j]
        
        if rem_cap_after_j > 0:
            # Filter unvisited_arr for k != j
            k_candidates_mask = (unvisited_arr != j) & (demands_arr[unvisited_arr] <= rem_cap_after_j)
            
            if np.any(k_candidates_mask):
                k_candidates = unvisited_arr[k_candidates_mask]
                
                # Cost from j to k + k to depot
                # d(j, k) + d(k, depot)
                d_jk = dm[j, k_candidates]
                d_kd = dm[k_candidates, depot]
                
                # Minimize d_jk + d_kd
                min_k_cost = np.min(d_jk + d_kd)
                
                # Choose the better option: go to k or return to depot
                if min_k_cost < min_future_cost:
                    min_future_cost = min_k_cost
        
        total_score = cost_to_j + min_future_cost
        
        if total_score < best_score:
            best_score = total_score
            best_node = int(j)
            
    return best_node
