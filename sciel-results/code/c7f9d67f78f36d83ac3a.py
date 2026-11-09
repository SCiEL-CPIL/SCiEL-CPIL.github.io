"""Guided local search penalty for TSP: 2-opt move regret-based penalties."""
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 320.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    if time.perf_counter() > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()

    n = edge_distance.shape[0]
    if n <= 2:
        return np.asarray(edge_distance, dtype=float).copy()

    # True distances (always input is the true matrix)
    d = np.asarray(edge_distance, dtype=float)

    # Build the edge list of the current tour (closed cycle)
    tour = np.asarray(local_opt_tour, dtype=int)
    if tour.shape[0] != n:
        tour = np.arange(n)

    # Edge endpoints for the closed tour
    i = tour[:-1]
    j = tour[1:]
    i = np.append(i, tour[0])
    j = np.append(j, tour[-1])

    # Lengths of each tour edge (true)
    tour_len = d[i, j]
    mean_tour_len = np.mean(tour_len)

    # Usage-based penalty for exploration
    usage = np.asarray(edge_n_used, dtype=float)
    usage = np.maximum(usage, 0.0)
    u_max = usage.max()
    if u_max > 0:
        # Use log scaling to dampen the effect of very high counts
        usage_scaled = np.log1p(usage) / np.log1p(u_max)
    else:
        usage_scaled = np.zeros_like(usage)
        
    # Extract usage counts for the current tour edges
    edge_usage = usage_scaled[i, j]
    
    # Strategy: 
    # 1. Penalize all tour edges to encourage change (escape local optimum).
    # 2. Increase penalty for edges that have been used frequently in previous iterations.
    # 3. Optionally, penalize shorter edges more heavily to break the "core" structure of the local optimum.
    
    # Base penalty: proportional to the edge length? No, let's use a fixed scale relative to mean tour length.
    # This ensures the penalty is significant compared to the distances.
    base_penalty = 0.1 * mean_tour_len
    
    # Usage penalty: up to 0.5 * mean_tour_len for the most used edges
    usage_penalty = 0.5 * mean_tour_len * edge_usage
    
    # Structure penalty: penalize edges that are shorter than the median tour edge.
    # This targets the "strong" links that define the local optimum.
    median_len = np.median(tour_len)
    # If edge is shorter than median, penalty increases.
    # Max penalty for shortest edge: 0.3 * mean_tour_len
    struct_penalty = 0.3 * mean_tour_len * np.clip((median_len - tour_len) / (median_len + 1e-9), 0, 1)
    
    total_penalty = base_penalty + usage_penalty + struct_penalty
    
    # Apply penalties to the distance matrix
    penalty = np.zeros((n, n), dtype=float)
    penalty[i, j] = total_penalty
    penalty[j, i] = total_penalty
    
    # Symmetrize (just in case, though we set both)
    penalty = 0.5 * (penalty + penalty.T)
    
    # Result: true distance + penalty
    result = d + penalty
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)
    
    return result
