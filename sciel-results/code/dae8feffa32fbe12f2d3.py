"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = None

def _get_deadline():
    global _deadline
    if _deadline is None:
        _deadline = time.time() + 110.0
    return _deadline

def _time_exceeded():
    return time.time() > _get_deadline()

def _fast_greedy_completion(start_node, depot, unvisited_list, rest_capacity, demands, dm):
    """
    Fast greedy heuristic to estimate the cost of completing a route from start_node.
    Uses savings-based scoring for selection.
    Returns the total distance of the completed route segment (start_node -> ... -> depot).
    """
    total_dist = 0.0
    current = start_node
    remaining_cap = rest_capacity
    remaining = list(unvisited_list)
    
    # Pre-compute distances from depot to all remaining for savings calculation
    # dm_depot = dm[depot, remaining] # Not strictly needed if we compute on fly
    
    while remaining:
        # Filter feasible nodes
        feasible = [n for n in remaining if demands[n] <= remaining_cap]
        if not feasible:
            break
        
        if len(feasible) == 1:
            next_node = feasible[0]
        else:
            feas_arr = np.array(feasible)
            d_cur = dm[current, feas_arr]
            d_depot = dm[feas_arr, depot]
            
            # Savings: d(cur, depot) + d(depot, i) - d(cur, i)
            # We want to maximize savings, so subtract it from distance
            savings = dm[current, depot] + d_depot - d_cur
            
            # Adaptive lambda: 
            # If capacity is tight, we should be more conservative (lower lambda) 
            # to avoid getting stuck far from depot.
            # If capacity is ample, higher lambda to maximize savings.
            # Estimate remaining nodes possible
            avg_demand = np.mean(demands[feas_arr]) if len(feas_arr) > 0 else 1.0
            slots = remaining_cap / max(avg_demand, 1.0)
            
            if slots < 3.0:
                lam = 0.2
            elif slots < 6.0:
                lam = 0.5
            else:
                lam = 0.8
                
            scores = d_cur - lam * savings
            best_idx = np.argmin(scores)
            next_node = int(feas_arr[best_idx])
            
        total_dist += dm[current, next_node]
        current = next_node
        remaining_cap -= demands[next_node]
        remaining.remove(next_node)
        
    total_dist += dm[current, depot]
    return total_dist

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a hybrid of savings-based scoring and greedy lookahead.
    
    1. If the number of feasible candidates is small (<= 15), we use a 1-step lookahead:
       We simulate the completion of the route for each candidate using a fast greedy
       heuristic and pick the candidate that yields the lowest total estimated route cost.
       
    2. Otherwise, we use a vectorized savings-based score with capacity-aware weighting.
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    if _time_exceeded():
        # Emergency fallback: nearest feasible node
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])
    
    # Identify feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Structural Change: Use Lookahead Simulation for small candidate sets
    # This is significantly more accurate than a linear score for the end of routes
    if len(feasible) <= 15:
        best_cost = float('inf')
        best_node = feasible[0]
        
        # Pre-compute distances from current to all feasible
        feas_arr = np.array(feasible)
        d_cur_to_feas = distance_matrix[current_node, feas_arr]
        
        for i, cand in enumerate(feasible):
            # Cost to go to candidate
            cost_to_cand = d_cur_to_feas[i]
            
            # Estimate cost of remaining route from candidate
            # We pass the list of unvisited nodes excluding the candidate
            # Note: This is O(N) to create the list, but N is small here
            remaining_unvisited = [n for n in unvisited_nodes if n != cand]
            
            # Simulate completion
            est_remaining_cost = _fast_greedy_completion(
                cand, depot, remaining_unvisited, 
                rest_capacity - demands[cand], 
                demands, distance_matrix
            )
            
            total_est_cost = cost_to_cand + est_remaining_cost
            
            if total_est_cost < best_cost:
                best_cost = total_est_cost
                best_node = cand
        
        return int(best_node)
    
    # For larger sets, use vectorized savings-based scoring
    feas_arr = np.array(feasible)
    
    # Distances
    d_cur = distance_matrix[current_node, feas_arr]
    d_depot = distance_matrix[feas_arr, depot]
    
    # Savings calculation
    # saving(i) = d(current, depot) + d(depot, i) - d(current, i)
    # Minimizing d(cur, i) - lambda * saving(i)
    # = d(cur, i) - lambda * (d(cur, depot) + d(depot, i) - d(cur, i))
    # = (1 + lambda) * d(cur, i) - lambda * d(cur, depot) - lambda * d(depot, i)
    
    # Capacity-aware lambda
    avg_demand = np.mean(demands[feas_arr])
    remaining_slots = rest_capacity / max(avg_demand, 1.0)
    
    if remaining_slots < 3.0:
        lam = 0.3
    elif remaining_slots < 6.0:
        lam = 0.6
    else:
        lam = 0.9
        
    d_cur_depot = distance_matrix[current_node, depot]
    
    scores = d_cur - lam * (d_cur_depot + d_depot - d_cur)
    
    best_idx = np.argmin(scores)
    return int(feas_arr[best_idx])

# EVOLVE-BLOCK-END
