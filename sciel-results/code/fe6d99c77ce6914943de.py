"""Savings-based heuristic for Capacitated Vehicle Routing Problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a combined savings and nearest-neighbor heuristic.
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

    # 2. Calculate savings score and nearest neighbor score
    # Savings = dist(current, depot) + dist(depot, candidate) - dist(current, candidate)
    # High savings means it's better to visit the candidate now than to return to depot and start a new route.
    
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_cand = distance_matrix[depot, feasible]
    d_cur_cand = distance_matrix[current_node, feasible]
    
    savings = d_cur_dep + d_dep_cand - d_cur_cand
    
    # Nearest neighbor score: we want to minimize distance, so we can use negative distance
    # or just normalize both scores to [0, 1] and combine
    dist_to_cur = d_cur_cand
    
    # Normalize scores to [0, 1] range for blending
    # For savings: higher is better
    # For distance: lower is better, so we use -dist or 1/dist
    
    # Handle edge cases for normalization
    savings_min = np.min(savings)
    savings_max = np.max(savings)
    dist_min = np.min(dist_to_cur)
    dist_max = np.max(dist_to_cur)
    
    # If all values are the same, avoid division by zero
    if savings_max == savings_min:
        norm_savings = np.zeros_like(savings, dtype=float)
    else:
        norm_savings = (savings - savings_min) / (savings_max - savings_min)
    
    if dist_max == dist_min:
        norm_dist = np.zeros_like(dist_to_cur, dtype=float)
    else:
        # Invert distance so that closer nodes have higher normalized scores
        norm_dist = (dist_max - dist_to_cur) / (dist_max - dist_min)
    
    # 3. Blend the two scores
    # Based on findings, savings-based (prioritizing far nodes) improved scores significantly.
    # But pure savings can lead to zig-zag patterns. Blending with nearest neighbor should help.
    # Let's use alpha = 0.7 for savings, 0.3 for nearest neighbor
    alpha = 0.7
    combined_score = alpha * norm_savings + (1 - alpha) * norm_dist
    
    best_idx = np.argmax(combined_score)
    
    return int(feasible[best_idx])

# EVOLVE-BLOCK-END
