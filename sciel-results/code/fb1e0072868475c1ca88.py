"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 320.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Check time limit - return original distances if we're running out of time
    if time.time() - _GLOBAL_START_TIME > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # If no edges have been used yet, return original
    if edge_n_used.max() == 0:
        return edge_distance.copy()
    
    # Compute average edge length of the current tour for scale awareness
    tour_len = 0.0
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_len += edge_distance[i, k]
    avg_edge_len = tour_len / n if n > 0 else 1.0
    
    # Extract tour edges as a boolean mask
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_edges[i, j] = True
        tour_edges[j, i] = True
    
    # Estimate round progress from max usage count
    # Over 20 rounds, max count grows from ~1 to ~20
    max_count = float(np.max(edge_n_used))
    if max_count <= 0:
        max_count = 1.0
    
    # Round progress: 0 at start, 1 at end (approximate)
    # Using a smooth sigmoid-like mapping
    round_progress = min(1.0, max_count / 15.0)  # ~1.0 by round 15+
    
    # Adaptive penalty coefficients that increase with round progress
    # Early rounds: gentle perturbation to explore
    # Later rounds: stronger pressure to escape local optima
    base_alpha_tour = 0.015 * avg_edge_len
    base_alpha_global = 0.003 * avg_edge_len
    
    # Scale up penalties as we progress through rounds
    # Use a smooth increase: alpha * (1 + round_progress * boost_factor)
    boost_factor = 2.0
    alpha_tour = base_alpha_tour * (1.0 + round_progress * boost_factor)
    alpha_global = base_alpha_global * (1.0 + round_progress * boost_factor)
    
    # Use a smooth, saturating function of usage counts instead of linear
    # This avoids excessive penalty on very high-count edges while still
    # providing meaningful pressure
    # f(count) = count / (1 + count * smoothing_param)
    # This is bounded and smooth, unlike raw linear scaling
    smoothing_param = 0.1
    smooth_counts = edge_n_used / (1.0 + smoothing_param * edge_n_used)
    
    # Build penalty matrix with smooth count scaling
    # Tour edges get stronger coefficient, others get weaker
    penalty = np.where(tour_edges, alpha_tour, alpha_global) * smooth_counts
    
    # Additive penalty on top of true distances
    penalized = edge_distance + penalty
    
    # Ensure finiteness and positivity
    penalized = np.where(np.isfinite(penalized), penalized, edge_distance)
    penalized = np.maximum(penalized, 1e-10)
    
    # Symmetrize to guard against any numerical drift
    penalized = (penalized + penalized.T) / 2.0
    
    return penalized
# EVOLVE-BLOCK-END
