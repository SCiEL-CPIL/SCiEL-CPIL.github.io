"""Model-and-optimize heuristic for Capacitated VRP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Improved heuristic with 2-step lookahead:
    For each feasible candidate, estimate the total cost of:
      current -> candidate -> (best next feasible or depot)
    This captures synergy/anti-synergy between consecutive visits.
    """
    # Time guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: nearest feasible node
        unvisited_arr = np.asarray(unvisited_nodes)
        dm_arr = np.asarray(distance_matrix)
        dems_arr = np.asarray(demands)
        feasible = unvisited_arr[dems_arr[unvisited_arr] <= rest_capacity]
        if len(feasible) == 0:
            return 0
        return int(feasible[np.argmin(dm_arr[current_node, feasible])])

    if len(unvisited_nodes) == 0:
        return 0

    unvisited = np.asarray(unvisited_nodes)
    dm = np.asarray(distance_matrix)
    dems = np.asarray(demands)

    # Filter to feasible customers
    feasible = unvisited[dems[unvisited] <= rest_capacity]
    if len(feasible) == 0:
        return 0

    n_feasible = len(feasible)
    
    if n_feasible == 1:
        return int(feasible[0])

    # Distances from current node to each feasible
    dist_cur = dm[current_node, feasible]
    
    # Distances back to depot
    dist_depot = dm[feasible, depot]
    
    # Demand of each feasible customer
    reqs = dems[feasible]
    
    # 2-step lookahead: for each candidate i, find the best next node j
    # (or depot) to visit after i, considering remaining capacity
    # Cost estimate: dist(cur,i) + min_j [dist(i,j) + dist(j,depot)] or dist(i,depot)
    lookahead_costs = np.zeros(n_feasible)
    
    for i in range(n_feasible):
        node_i = feasible[i]
        remaining_cap = rest_capacity - reqs[i]
        
        # Options: go to depot, or go to another feasible node that fits
        best_next_cost = dm[node_i, depot]  # default: return to depot
        
        if remaining_cap > 0:
            # Find other feasible nodes that would fit after visiting i
            other_feasible = unvisited[dems[unvisited] <= remaining_cap]
            # Exclude node_i itself
            mask = other_feasible != node_i
            others = other_feasible[mask]
            
            if len(others) > 0:
                # For each other node j, cost = dist(i,j) + dist(j,depot)
                dists_i_j = dm[node_i, others]
                dists_j_depot = dm[others, depot]
                next_costs = dists_i_j + dists_j_depot
                best_next_cost = min(best_next_cost, np.min(next_costs))
        
        lookahead_costs[i] = dist_cur[i] + best_next_cost

    # Also compute a simple score for tie-breaking and additional penalties
    # Capacity usage fraction
    cap_frac = reqs / rest_capacity if rest_capacity > 0 else np.ones_like(reqs, dtype=float)
    
    # Final score: primarily lookahead cost, with small adjustments
    # Penalize high capacity usage slightly to encourage balanced routes
    score = lookahead_costs + 0.05 * cap_frac * dist_depot

    best_idx = int(np.argmin(score))
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
