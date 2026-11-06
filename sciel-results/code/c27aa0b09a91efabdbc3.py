"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 148.0  # seconds, with 2s safety margin

def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: multi-criteria scoring with lookahead regret.
    
    Strategy:
    1. Filter unvisited nodes that fit in rest_capacity.
    2. If none fit, return depot (0).
    3. For each feasible node, compute a composite score:
       a. Immediate distance from current_node (lower is better)
       b. Lookahead: minimum distance from candidate to any other feasible node or depot
          (penalize nodes that are close now but far from all remaining options)
       c. Capacity utilization: prefer nodes with higher demand to fill capacity
    4. Return the node with the best (lowest) composite score.
    """
    if _time_exceeded():
        # Fallback: nearest feasible node
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])
    
    n = len(unvisited_nodes)
    if n == 0:
        return 0
    
    # Filter feasible nodes
    feasible = []
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            feasible.append(node)
    
    if not feasible:
        return 0
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible)
    dist_to_feasible = distance_matrix[current_node, feasible_arr]
    
    # Compute lookahead: for each feasible node, min distance to other feasible nodes or depot
    # Use vectorized computation
    # dist_matrix[feasible_arr, :].min(axis=1) gives min distance from each feasible to any node
    # But we want min distance to other feasible nodes OR depot, excluding self
    lookahead = np.empty(len(feasible_arr))
    for i, node in enumerate(feasible_arr):
        # Min distance to all other feasible nodes (excluding self) and to depot
        if len(feasible_arr) > 1:
            other_indices = np.concatenate([np.arange(i), np.arange(i+1, len(feasible_arr))])
            min_to_other = distance_matrix[node, feasible_arr[other_indices]].min()
        else:
            min_to_other = distance_matrix[node, depot]
        min_to_depot = distance_matrix[node, depot]
        lookahead[i] = min(min_to_other, min_to_depot)
    
    # Capacity utilization: higher demand is better (use negative demand as penalty)
    cap_util = -demands[feasible_arr].astype(float)
    
    # Composite score: weighted sum
    # Normalize each component to [0, 1]
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_dist = normalize(dist_to_feasible)
    norm_lookahead = normalize(lookahead)
    norm_cap = normalize(cap_util)
    
    # Weights: immediate distance most important, lookahead second, capacity third
    weights = np.array([0.5, 0.35, 0.15])
    scores = weights[0] * norm_dist + weights[1] * norm_lookahead + weights[2] * norm_cap
    
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
