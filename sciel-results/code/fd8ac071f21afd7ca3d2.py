"""Rewritten TSP constructive heuristic with improved lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: total wall-clock budget for all calls combined.
_DEADLINE = 115.0
_START_TIME = time.time()

# Cache for Held-Karp results to avoid recomputation within same tour
_hk_cache = {}


def _elapsed():
    return time.time() - _START_TIME


def _time_exceeded():
    return _elapsed() > _DEADLINE


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Picks the next city in a TSP tour.
    
    Strategy:
    - If there are few unvisited nodes (<= 12), use Held-Karp DP for exact optimal suffix.
    - Otherwise, use an improved lookahead heuristic that combines:
      * Immediate edge cost
      * Return-to-destination penalty
      * Nearest-neighbor simulation of the remaining tour
    
    We also use a time guard: if we're running low on time, fall back to nearest neighbor.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return current_node
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is running out, use simple nearest neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For small sets, use exact Held-Karp
    if n_unvisited <= 12:
        return _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # For larger sets, use improved lookahead heuristic
    return _lookahead_select(current_node, destination_node, unvisited_nodes, distance_matrix)


def _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Use Held-Karp DP to find the optimal next node.
    Uses caching to avoid recomputation.
    """
    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Create a cache key based on the sorted tuple of unvisited nodes
    sorted_nodes = tuple(sorted(unvisited_nodes))
    cache_key = (current_node, destination_node, sorted_nodes)
    
    if cache_key in _hk_cache:
        return _hk_cache[cache_key]
    
    node_list = unvisited_nodes.astype(int)
    
    best_cost = np.inf
    best_next = unvisited_nodes[0]
    
    for j_idx in range(n):
        j = node_list[j_idx]
        remaining = np.delete(node_list, j_idx)
        
        if len(remaining) == 0:
            cost = distance_matrix[current_node, j] + distance_matrix[j, destination_node]
        else:
            m = len(remaining)
            
            if m == 1:
                cost = (distance_matrix[current_node, j] + 
                       distance_matrix[j, remaining[0]] + 
                       distance_matrix[remaining[0], destination_node])
            else:
                # Held-Karp for path from j through remaining to destination
                dp = np.full((1 << m, m), np.inf)
                for k in range(m):
                    dp[1 << k, k] = distance_matrix[j, remaining[k]]
                
                for mask in range(1, 1 << m):
                    for k in range(m):
                        if not (mask & (1 << k)):
                            continue
                        if mask == (1 << k):
                            continue
                        prev_mask = mask ^ (1 << k)
                        for l in range(m):
                            if prev_mask & (1 << l):
                                new_cost = dp[prev_mask, l] + distance_matrix[remaining[l], remaining[k]]
                                if new_cost < dp[mask, k]:
                                    dp[mask, k] = new_cost
                
                full_mask = (1 << m) - 1
                # Find the best ending node
                min_suffix = np.min(dp[full_mask, :])
                # Find which node achieves this minimum
                best_end_idx = np.argmin(dp[full_mask, :])
                cost = (distance_matrix[current_node, j] + 
                       min_suffix + 
                       distance_matrix[remaining[best_end_idx], destination_node])
        
        if cost < best_cost:
            best_cost = cost
            best_next = j
    
    _hk_cache[cache_key] = int(best_next)
    return int(best_next)


def _lookahead_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Improved lookahead heuristic: for each candidate, estimate the cost of going to it
    plus a fast estimate of the remaining tour cost using nearest-neighbor simulation
    with a return-to-destination penalty.
    """
    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Precompute distances for efficiency
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    best_cost = np.inf
    best_next = unvisited_nodes[0]
    
    for j_idx in range(n):
        j = unvisited_nodes[j_idx]
        remaining = np.delete(unvisited_nodes, j_idx)
        
        # Estimate: nearest neighbor path from j through remaining to destination
        current_pos = j
        est_cost = dist_from_current[j_idx]
        
        # Simulate nearest-neighbor tour through remaining nodes
        remaining_set = set(remaining.tolist())
        for _ in range(len(remaining)):
            if not remaining_set:
                break
            # Find nearest unvisited node using vectorized numpy
            candidates = np.array(list(remaining_set))
            distances = distance_matrix[current_pos, candidates]
            nearest_idx = np.argmin(distances)
            nearest = candidates[nearest_idx]
            est_cost += distances[nearest_idx]
            current_pos = nearest
            remaining_set.remove(nearest)
        
        # Add return to destination
        if current_pos != destination_node:
            est_cost += distance_matrix[current_pos, destination_node]
        
        if est_cost < best_cost:
            best_cost = est_cost
            best_next = j
    
    return int(best_next)
# EVOLVE-BLOCK-END
