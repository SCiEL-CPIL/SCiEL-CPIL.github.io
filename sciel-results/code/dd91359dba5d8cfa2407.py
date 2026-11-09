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
    
    # Distances from current to feasible nodes
    dists_curr = np.asarray(distance_matrix[current_node, feasible_nodes], dtype=np.float64)
    
    # Soft-min distance from each feasible node to other feasible nodes
    # Use log-sum-exp with a temperature for smooth surrogate
    sub_matrix = np.asarray(distance_matrix[np.ix_(feasible_nodes, feasible_nodes)], dtype=np.float64)
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Soft-min: -T * log(mean(exp(-d/T))) approximates min but is smooth
    # Use T=1.0 for reasonable smoothing
    T = 1.0
    # For numerical stability, subtract max before exp
    min_dists = np.min(sub_matrix, axis=1)
    # Soft-min: logsumexp(-d/T) / (-1/T) = -T * logsumexp(-d/T)
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    neg_d_over_T = -sub_matrix / T
    max_neg = np.max(neg_d_over_T, axis=1, keepdims=True)
    # Handle inf: where all are inf, result should be inf
    log_sum_exp = max_neg.squeeze() + np.log(np.sum(np.exp(neg_d_over_T - max_neg), axis=1))
    soft_min = -T * log_sum_exp
    # Replace any nan/inf with a large value
    soft_min = np.where(np.isfinite(soft_min), soft_min, 1e6)
    
    # Depot return penalty: scaled by how "full" the route is
    # When capacity is low, we should prefer nodes closer to depot
    dists_depot = np.asarray(distance_matrix[feasible_nodes, depot], dtype=np.float64)
    
    # Estimate total capacity from demands (use max demand as proxy, or sum)
    # Use rest_capacity relative to a reasonable scale
    # Scale: if rest_capacity is small, penalize far-from-depot nodes more
    cap_scale = max(rest_capacity, 1)
    depot_weight = 0.3 * (1.0 - min(rest_capacity, 50) / 50.0)
    
    alpha = 0.4  # Weight for cluster proximity (soft-min)
    scores = dists_curr + alpha * soft_min + depot_weight * dists_depot
    
    # Deadline guard
    if time.time() - t_start > 145:
        min_idx = np.argmin(dists_curr)
        return int(feasible_nodes[min_idx])
    
    min_idx = np.argmin(scores)
    return int(feasible_nodes[min_idx])
