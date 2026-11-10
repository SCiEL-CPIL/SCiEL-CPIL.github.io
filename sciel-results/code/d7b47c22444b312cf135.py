"""Structural redesign: regret-based + capacity-aware lookahead heuristic for CVRP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    # Deadline guard: if we're close to the limit, just pick nearest feasible
    if _elapsed() > 110.0:
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return 0
        return int(min(feasible, key=lambda n: distance_matrix[current_node, n]))

    if not unvisited_nodes:
        return 0

    # Filter to feasible nodes (capacity constraint)
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible:
        return 0

    # If only one feasible, take it
    if len(feasible) == 1:
        return int(feasible[0])

    # Compute distances from current to all feasible nodes
    dists = np.array([distance_matrix[current_node, n] for n in feasible])
    
    # Regret-based scoring: 
    # For each feasible node i, compute the "regret" of NOT visiting it now:
    # regret_i = (second_best distance from current to any feasible node) - (best distance)
    # Actually, a better formulation: for each node, what is the cost if we skip it?
    # We approximate: the node with the smallest distance has low regret (we can come back),
    # but the node that is far from ALL others has high regret (if we skip it, it becomes isolated).
    
    # Compute for each feasible node, its minimum distance to any OTHER feasible node (or depot)
    # This represents "how hard would it be to reach later?"
    n_feasible = len(feasible)
    feasible_arr = np.array(feasible)
    
    # For each feasible node, find its best connection to another feasible node or depot
    # This is a proxy for "isolation cost" - if a node is far from all others, visiting it now is important
    min_dist_to_other = np.zeros(n_feasible)
    for i in range(n_feasible):
        node_i = feasible_arr[i]
        # Distance to all other feasible nodes and to depot
        other_indices = np.concatenate([feasible_arr, [depot]])
        other_indices = other_indices[other_indices != node_i]
        if len(other_indices) > 0:
            min_dist_to_other[i] = np.min(distance_matrix[node_i, other_indices])
        else:
            min_dist_to_other[i] = 0.0
    
    # Regret score: nodes that are far from others (high isolation) should be prioritized
    # because if we skip them, they become harder to reach
    # Combined with direct distance from current
    
    # Normalize distances for weighting
    max_dist = np.max(dists) if np.max(dists) > 0 else 1.0
    max_isolation = np.max(min_dist_to_other) if np.max(min_dist_to_other) > 0 else 1.0
    
    # Score: lower is better
    # Direct distance component
    direct_score = dists / max_dist
    # Isolation component: we want to visit isolated nodes sooner
    isolation_score = min_dist_to_other / max_isolation
    
    # Combine: prefer nodes that are close AND/OR isolated
    # Weight: 0.6 * direct + 0.4 * isolation
    combined = 0.6 * direct_score + 0.4 * isolation_score
    
    # Capacity-aware lookahead: if rest_capacity is getting low, prefer smaller demand nodes
    # to allow more customers to be served
    if rest_capacity < 30:
        demand_penalty = np.array([demands[n] for n in feasible]) / 9.0
        combined = combined + 0.15 * demand_penalty
    
    # Pick the node with lowest combined score
    best_idx = int(np.argmin(combined))
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
