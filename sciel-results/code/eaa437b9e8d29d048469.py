"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 148.0
_SAFETY_MARGIN = 2.0


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated vehicle routing.
    Picks the next customer to visit, or returns depot (0) to close the route.
    
    Strategy:
    1. Filter unvisited nodes that fit within rest_capacity.
    2. If none fit, return depot.
    3. Score each feasible candidate using a multi-criteria heuristic:
       - Immediate distance from current_node to candidate
       - 2-step lookahead: min distance from candidate to any other feasible node
       - Penalty for nodes that are far from the depot (to encourage early return
         if the route is getting long)
       - Bias towards nodes that are closer to the centroid of remaining nodes
         (to cluster visits)
    """
    if _time_exceeded():
        # Safety: return nearest feasible or depot
        if len(unvisited_nodes) == 0:
            return 0
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])
    
    if len(unvisited_nodes) == 0:
        return 0
    
    # Filter feasible nodes
    feasible_mask = np.array([demands[n] <= rest_capacity for n in unvisited_nodes])
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0
    
    # If only one feasible node, take it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Compute scores for each feasible candidate
    # We want to minimize the score
    
    # 1. Immediate distance from current_node
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    
    # 2. Lookahead: min distance from candidate to any other feasible node (or depot if we might return)
    #    This encourages choosing nodes that are well-connected to other unvisited nodes
    lookahead_scores = np.zeros(len(feasible_nodes))
    for i, cand in enumerate(feasible_nodes):
        # Min distance from candidate to other feasible nodes (excluding self)
        if len(feasible_nodes) > 1:
            other_mask = np.ones(len(feasible_nodes), dtype=bool)
            other_mask[i] = False
            other_nodes = feasible_nodes[other_mask]
            min_dist_to_others = np.min(distance_matrix[cand, other_nodes])
        else:
            min_dist_to_others = distance_matrix[cand, depot]
        lookahead_scores[i] = min_dist_to_others
    
    # 3. Distance from candidate to depot (penalize if we're likely to have to return soon)
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # 4. Centroid-based clustering: compute centroid of all unvisited nodes
    #    and prefer candidates close to the centroid
    if len(unvisited_nodes) > 0:
        # Use 2D coordinates? No, we only have distance matrix.
        # Instead, use a proxy: average distance from candidate to all other unvisited nodes
        # Nodes with lower average distance to others are more "central"
        avg_dist_to_others = np.zeros(len(feasible_nodes))
        for i, cand in enumerate(feasible_nodes):
            avg_dist_to_others[i] = np.mean(distance_matrix[cand, unvisited_nodes])
    else:
        avg_dist_to_others = np.zeros(len(feasible_nodes))
    
    # Normalize each component to [0, 1] range for weighting
    def normalize(arr):
        if np.max(arr) - np.min(arr) < 1e-10:
            return np.zeros_like(arr)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    norm_dist_current = normalize(dist_from_current)
    norm_lookahead = normalize(lookahead_scores)
    norm_depot = normalize(dist_to_depot)
    norm_centroid = normalize(avg_dist_to_others)
    
    # Weighted sum: we want to minimize
    # - Immediate distance (weight 1.0)
    # - Lookahead connectivity (weight 0.5) - lower is better (well-connected)
    # - Distance to depot (weight 0.3) - lower is better (stay close to depot if capacity is tight)
    # - Centrality (weight 0.5) - lower is better (central nodes)
    
    # Adjust depot weight based on remaining capacity
    total_remaining_demand = sum(demands[n] for n in unvisited_nodes)
    if rest_capacity < 0.5 * (total_remaining_demand + rest_capacity):
        depot_weight = 0.8  # High weight if capacity is tight
    else:
        depot_weight = 0.3
    
    score = (1.0 * norm_dist_current + 
             0.5 * norm_lookahead + 
             depot_weight * norm_depot + 
             0.5 * norm_centroid)
    
    best_idx = np.argmin(score)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
