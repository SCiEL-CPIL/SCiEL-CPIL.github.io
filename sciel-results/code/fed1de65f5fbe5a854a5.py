"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE_MARGIN = 2.0  # seconds before hard kill


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: pick next customer or depot (0) to close a route.
    
    Strategy:
      1. If no unvisited nodes, return depot.
      2. Filter to feasible nodes (demand fits in rest_capacity).
      3. If no feasible nodes, return depot.
      4. Score each feasible node using:
         - Immediate distance from current_node
         - 2-step lookahead: for each candidate, estimate min cost to go to another 
           feasible node (or depot) and then to the next best (or depot)
         - Capacity-based penalty to encourage packing
      5. Return the best-scoring node.
    """
    # Deadline guard
    if time.time() - _start_time > 120.0 - _DEADLINE_MARGIN:
        # Fallback: nearest feasible or depot
        if len(unvisited_nodes) == 0:
            return depot
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    if len(unvisited_nodes) == 0:
        return depot

    # Filter feasible nodes
    feasible = []
    for n in unvisited_nodes:
        if demands[n] <= rest_capacity:
            feasible.append(n)
    
    if not feasible:
        return depot

    feasible = np.array(feasible)
    
    # If only one feasible node, take it
    if len(feasible) == 1:
        return int(feasible[0])

    # Compute scores
    # Immediate distance from current_node
    imm_dist = distance_matrix[current_node, feasible]
    
    # 2-step lookahead:
    # For each candidate i, we estimate:
    #   step1[i] = min over j (j != i) of dist[feasible[i], feasible[j]] 
    #   step2[i] = min over k (k != i, k != argmin_j) of dist[feasible[j_best], feasible[k]] or dist[feasible[j_best], depot]
    # 
    # A simpler but effective 2-step:
    #   For each candidate i:
    #     best_next = argmin_j dist[i,j] (j in feasible, j != i)
    #     lookahead2[i] = dist[i, best_next] + min( dist[best_next, k] for k in feasible, k != i, k != best_next, 
    #                                               dist[best_next, depot] )
    #
    # This is O(n^2) per candidate which is O(n^3) total. For 50 nodes this is fine.
    # But we can vectorize partially.
    
    # Distance matrix among feasible nodes
    dist_feas = distance_matrix[np.ix_(feasible, feasible)]
    n = len(feasible)
    
    # For each i, find best next (excluding self)
    # Set diagonal to inf
    dist_feas_copy = dist_feas.copy()
    np.fill_diagonal(dist_feas_copy, np.inf)
    
    # For each row i, best next index
    best_next_idx = np.argmin(dist_feas_copy, axis=1)
    best_next_dist = np.min(dist_feas_copy, axis=1)
    
    # For the second step, for each i, given best_next_idx[i], we want:
    #   min( dist[feasible[best_next_idx[i]], k] for k in feasible, k != i, k != best_next_idx[i]
    #        dist[feasible[best_next_idx[i]], depot] )
    #
    # Let's compute this.
    # For each i, let j = best_next_idx[i]
    # We want min of dist_feas[j, :] with elements at positions i and j set to inf, 
    # and also compare with dist_to_depot[j]
    
    # Create a masked distance matrix for the second step
    # second_step[i] = min( min(dist_feas[best_next_idx[i], k] for k not in {i, best_next_idx[i]}), 
    #                       dist_to_depot[best_next_idx[i]] )
    
    dist_to_depot = distance_matrix[feasible, depot]
    
    second_step = np.zeros(n)
    for i in range(n):
        j = best_next_idx[i]
        # Get distances from j to all feasible nodes
        d_j = dist_feas[j, :].copy()
        # Mask out i and j
        d_j[i] = np.inf
        d_j[j] = np.inf
        min_second = np.min(d_j) if n > 2 else np.inf
        # Compare with going to depot
        second_step[i] = min(min_second, dist_to_depot[j])
    
    # Total 2-step lookahead cost
    lookahead2 = best_next_dist + second_step
    
    # Weighted score: immediate distance + weight * 2-step lookahead
    alpha = 0.4
    scores = imm_dist + alpha * lookahead2
    
    # Capacity-based penalty: penalize nodes that consume a large fraction of remaining capacity
    # This encourages packing similar-sized demands together and avoids wasting capacity
    cap_penalty = 0.1 * (demands[feasible] / max(rest_capacity, 1))
    scores += cap_penalty
    
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
