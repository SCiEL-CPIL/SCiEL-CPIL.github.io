"""Adaptive savings/nearest-neighbor heuristic for Capacitated Vehicle Routing Problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using an adaptive heuristic:
    - When rest_capacity is high: emphasize savings (build long coherent routes)
    - When rest_capacity is low: emphasize nearest-neighbor + demand fit (pack tightly)
    Returns 0 (depot) if no feasible node can be added or if time limit is reached.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    # Convert to numpy array for vectorized operations
    unvisited = np.array(unvisited_nodes)
    
    # 1. Filter by capacity
    feasible = unvisited[demands[unvisited] <= rest_capacity]
    
    if len(feasible) == 0:
        return 0

    # Compute base distances
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_cand = distance_matrix[depot, feasible]
    d_cur_cand = distance_matrix[current_node, feasible]
    
    # Savings: how much we save by visiting candidate now vs. going back to depot
    savings = d_cur_dep + d_dep_cand - d_cur_cand
    
    # Distance to current node (lower is better)
    dist_to_cur = d_cur_cand
    
    # Demand ratio: how much of remaining capacity this node uses
    # We want a "good fit" - not too small (wastes capacity), not too large (leaves nothing)
    # Ideal is around 0.3-0.7 of remaining capacity for packing efficiency
    demand_vals = demands[feasible].astype(float)
    demand_ratio = demand_vals / rest_capacity
    
    # "Fit score": peaks around 0.5, penalizes very small or very large demands relative to capacity
    # Use a simple quadratic: fit = 1 - (demand_ratio - 0.5)^2, clamped to [0, 1]
    fit_score = np.clip(1.0 - (demand_ratio - 0.5) ** 2, 0.0, 1.0)
    
    # Normalize each component to [0, 1]
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr, dtype=float)
        return (arr - mn) / (mx - mn)
    
    norm_savings = normalize(savings)
    norm_dist_inv = 1.0 - normalize(dist_to_cur)  # closer = higher score
    # fit_score is already in [0, 1]
    
    # Adaptive weights based on remaining capacity fraction
    # Assume typical max capacity is around 50 (from problem statement)
    # Use rest_capacity directly as a proxy for "how full we are"
    # High rest_capacity -> we're early in the route -> emphasize savings
    # Low rest_capacity -> we're late in the route -> emphasize distance and fit
    
    # Normalize rest_capacity to [0, 1] assuming max ~50
    cap_fraction = np.clip(rest_capacity / 50.0, 0.0, 1.0)
    
    # When cap_fraction is high (early), weight savings heavily
    # When cap_fraction is low (late), weight distance and fit heavily
    w_savings = 0.3 + 0.5 * cap_fraction       # ranges from 0.3 to 0.8
    w_dist = 0.2 + 0.3 * (1.0 - cap_fraction)   # ranges from 0.5 to 0.2
    w_fit = 0.1 + 0.2 * (1.0 - cap_fraction)    # ranges from 0.3 to 0.1
    
    # Ensure weights sum to 1
    w_sum = w_savings + w_dist + w_fit
    w_savings /= w_sum
    w_dist /= w_sum
    w_fit /= w_sum
    
    combined_score = w_savings * norm_savings + w_dist * norm_dist_inv + w_fit * fit_score
    
    best_idx = np.argmax(combined_score)
    
    return int(feasible[best_idx])

# EVOLVE-BLOCK-END
