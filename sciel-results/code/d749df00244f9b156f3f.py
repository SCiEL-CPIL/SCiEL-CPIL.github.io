import numpy as np
import time

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    t_start = time.time()
    
    unvisited = np.array(unvisited_nodes)
    
    if len(unvisited) == 0:
        return int(depot)
        
    # Filter feasible nodes
    node_demands = np.array(demands)[unvisited]
    feasible_mask = node_demands <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
        
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    n_feas = len(feasible_nodes)
    
    # Distances from current to feasible nodes
    dists_curr = np.asarray(distance_matrix[current_node, feasible_nodes], dtype=np.float64)
    
    # Sub-matrix of distances between feasible nodes
    sub_matrix = np.asarray(distance_matrix[np.ix_(feasible_nodes, feasible_nodes)], dtype=np.float64)
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Min distance from each feasible node to any other feasible node
    min_exit_to_other = np.min(sub_matrix, axis=1)
    
    # Distances from feasible nodes to depot
    dists_depot = np.asarray(distance_matrix[feasible_nodes, depot], dtype=np.float64)
    
    # Capacity pressure: 0 when full capacity, 1 when empty
    # Assume max capacity is 50 as per task description
    max_cap = 50.0
    pressure = 1.0 - (rest_capacity / max_cap)
    
    # Weight for the "exit" cost
    # When pressure is high, we prioritize returning to depot.
    # We model the future cost as min(cost to next customer, cost to depot * penalty)
    # If we go to depot, we are done. If we go to customer, we continue.
    # To encourage returning when full, we make the depot option "cheaper" relative to customer?
    # No, we want to pick a node j such that the *total* route is short.
    # If we pick j, the next step is either k or depot.
    # The cost added by picking j is d(curr, j) + min( d(j, best_k), d(j, depot) * Weight )
    # If Weight is high, d(j, depot) is penalized, so we avoid nodes far from depot?
    # Wait. If we are full, we MUST return to depot. So the "future cost" is effectively d(j, depot).
    # If we are not full, the future cost is d(j, best_k).
    # So we can interpolate.
    
    # Let's define the effective exit cost
    # exit_cost = min(min_exit_to_other, dists_depot * (1.0 + pressure * 5.0))
    # This makes the depot option look more expensive when pressure is low (so we prefer customers)
    # and makes the depot option look competitive/expensive?
    # Actually, if pressure is high, we WANT to go to depot.
    # So the "cost" of going to depot should be low?
    # But d(j, depot) is fixed.
    # The decision is which j to pick.
    # If we pick j, we pay d(curr, j). Then we pay min(exit).
    # If we are full, we will pay d(j, depot).
    # If we are not full, we will pay d(j, k).
    # So, Score(j) = d(curr, j) + pressure * d(j, depot) + (1-pressure) * min_exit_to_other
    
    # This is a linear interpolation between "go to depot" and "go to cluster".
    # This seems robust.
    
    alpha = 1.0 # Weight for the future step
    
    scores = dists_curr + alpha * (pressure * dists_depot + (1.0 - pressure) * min_exit_to_other)
    
    # Deadline guard
    if time.time() - t_start > 145:
        min_idx = np.argmin(dists_curr)
        return int(feasible_nodes[min_idx])
    
    min_idx = np.argmin(scores)
    return int(feasible_nodes[min_idx])
