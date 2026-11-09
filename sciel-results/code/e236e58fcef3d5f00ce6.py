"""Guided local search penalty for TSP: Nearest Neighbor Bias Disruption."""
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 315.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Deadline guard
    if time.perf_counter() > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()

    n = edge_distance.shape[0]
    if n <= 2:
        return np.asarray(edge_distance, dtype=float).copy()

    # True distances
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

    # Identify K-Nearest Neighbors for each node
    # K=5 is a standard heuristic for TSP candidate edges
    K = 5
    nn_mask = np.zeros((n, n), dtype=bool)
    
    # Efficiently compute K-NN using argsort on rows
    # d has inf on diagonal? No, d is true distances. We should ignore self.
    d_sorted_idx = np.argsort(d, axis=1)
    
    # For each node, mark its K nearest neighbors
    for u in range(n):
        # Skip self if it appears in the first K (it shouldn't if d[u,u]=0 and others > 0)
        # But to be safe, we take the first K indices that are not u
        neighbors = d_sorted_idx[u, :K+1] # Take K+1 to allow for self-skip if needed
        count = 0
        for v in neighbors:
            if v == u:
                continue
            nn_mask[u, v] = True
            count += 1
            if count >= K:
                break

    # Symmetrize the NN mask: if v is in K-NN of u, or u is in K-NN of v
    # This defines the "Good Edge" set
    is_candidate = nn_mask | nn_mask.T

    # Build penalty matrix
    penalty = np.zeros((n, n), dtype=float)
    
    # Usage-based penalty: penalize edges that have been used many times
    usage = np.asarray(edge_n_used, dtype=float)
    usage = np.maximum(usage, 0.0)
    u_max = usage.max()
    
    # Base usage penalty: mild, to prevent cycling
    if u_max > 0:
        # Log-scaled usage
        usage_scaled = np.log1p(usage) / np.log1p(u_max)
        # Apply mild global usage penalty
        penalty += 0.1 * usage_scaled
    else:
        penalty += np.zeros_like(usage)

    # Structural Disruption:
    # Penalize edges in the current tour that are NOT candidate edges (not K-NN for either endpoint)
    # This forces the tour to shed "structurally unlikely" edges.
    
    tour_edge_indices = np.arange(n)
    
    # Vectorized check for candidate status of tour edges
    # is_candidate[i, j] is True if (i,j) is a candidate edge
    tour_is_candidate = is_candidate[i, j]
    
    # We want to penalize edges where tour_is_candidate is False
    # Penalty magnitude: proportional to edge length and usage
    
    tour_len = d[i, j]
    
    # Create a mask for non-candidate tour edges
    non_candidate_mask = ~tour_is_candidate
    
    # Apply penalty to non-candidate edges
    # We use a strong multiplier to ensure 2-opt breaks these edges
    # Penalty = alpha * length + beta * usage
    alpha = 0.5
    beta = 0.5
    
    # We only apply this to the specific edges in the tour
    # We can't just add to the matrix globally because we only want to disrupt the *current* bad edges
    # However, GLS usually adds penalty to the matrix for all edges, and the search finds a new tour.
    # If we only penalize the current tour's bad edges, the search might just swap them for other bad edges.
    # Better strategy: Penalize ALL non-candidate edges that are currently in the tour heavily.
    # Also, penalize non-candidate edges that have high usage (even if not in tour) to discourage future use.
    
    # 1. Heavy penalty on current tour edges that are non-candidates
    for idx in np.where(non_candidate_mask)[0]:
        a, b = i[idx], j[idx]
        # Magnitude scales with length and usage
        u_val = usage[a, b]
        mag = alpha * tour_len[idx] + beta * u_val
        penalty[a, b] += mag
        penalty[b, a] += mag

    # 2. Moderate penalty on non-candidate edges with high usage (global cleanup)
    # This helps clear out "bad" edges from the search space that might be reintroduced
    # Identify non-candidate edges
    non_candidate_edges = ~is_candidate
    # Exclude diagonal
    non_candidate_edges &= ~np.eye(n, dtype=bool)
    
    # Apply usage penalty only to non-candidates
    # penalty[non_candidate_edges] += gamma * usage[non_candidate_edges]
    # To do this efficiently without loops:
    if u_max > 0:
        # Only add extra usage penalty to non-candidate edges
        extra_usage_penalty = np.zeros((n, n), dtype=float)
        # We want: if not candidate, add 0.2 * usage_scaled
        # usage_scaled is already computed? No, let's reuse logic or recompute.
        # Let's just compute a specific matrix for this part
        usage_scaled = np.log1p(usage) / np.log1p(u_max)
        # Mask out candidates
        extra_usage_penalty[non_candidate_edges] = 0.2 * usage_scaled[non_candidate_edges]
        penalty += extra_usage_penalty

    # Symmetrize
    penalty = 0.5 * (penalty + penalty.T)

    # Result: true distance + penalty
    result = d + penalty

    # Ensure finiteness
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)

    return result
