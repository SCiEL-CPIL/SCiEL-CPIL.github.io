"""Guided local search penalty for TSP with model-and-optimize approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _two_opt(d, tour, max_passes=10):
    """Run 2-opt improvement on a tour using distance matrix d."""
    n = len(tour)
    tour = list(tour)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d2 = tour[j], tour[(j + 1) % n]
                # Check if 2-opt move improves
                if d[a][c] + d[b][d2] < d[a][b] + d[c][d2]:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty with optimized parameters.
    
    Uses a combination of:
    1. Direct edge usage penalty (standard GLS)
    2. Tour-proximity force field penalty via soft-min
    3. Additive penalty scaled by median edge length for uniform disruption
    
    Parameters are chosen to balance exploration and exploitation.
    """
    start_time = time.time()
    DEADLINE = 280  # seconds, safety margin before 300s limit
    
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()
    
    D = edge_distance
    
    # Extract tour edges
    tour_edges = []
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_edges.append((a, b))
    
    m = len(tour_edges)
    
    # Build arrays of tour edge endpoints
    a_arr = np.array([e[0] for e in tour_edges], dtype=int)
    b_arr = np.array([e[1] for e in tour_edges], dtype=int)
    
    # Compute soft-min closeness of each edge to any tour edge using log-sum-exp
    # For each tour edge (a, b), for all (i,j):
    #   closeness = min(D[i,a] + D[j,b], D[i,b] + D[j,a])
    # We want a smooth approximation of min over all tour edges
    
    # Temperature for soft-min: controls how close to true min
    # Smaller = closer to hard min, larger = smoother
    tau = 0.05  # scale relative to typical edge lengths (~0.1-0.3)
    
    # Accumulate log-sum-exp over all tour edges
    # log(sum_t exp(-c_t / tau)) = -soft_min(c) / tau
    log_sum_exp = np.full((n, n), -np.inf)
    
    for t in range(m):
        a = a_arr[t]
        b = b_arr[t]
        da = D[:, a]  # shape (n,)
        db = D[:, b]  # shape (n,)
        
        c1 = da[:, None] + db[None, :]  # D[i,a] + D[j,b]
        c2 = db[:, None] + da[None, :]  # D[i,b] + D[j,a]
        c = np.minimum(c1, c2)
        
        # Add to log-sum-exp in a numerically stable way
        # log_sum_exp = log(exp(log_sum_exp) + exp(-c/tau))
        # = max(log_sum_exp, -c/tau) + log(exp(min(...) - max(...)) + exp(max(...) - max(...)))
        term = -c / tau
        # Stable log-add: log(exp(a) + exp(b)) = max(a,b) + log(exp(a-max) + exp(b-max))
        mx = np.maximum(log_sum_exp, term)
        # Avoid log(0) when both are -inf
        with np.errstate(divide='ignore', invalid='ignore'):
            log_sum_exp = mx + np.log(np.exp(log_sum_exp - mx) + np.exp(term - mx))
        both_neg_inf = (log_sum_exp == -np.inf) & (term == -np.inf)
        log_sum_exp[both_neg_inf] = -np.inf
    
    # soft_min = -tau * log_sum_exp
    # Where log_sum_exp is -inf (shouldn't happen after first tour edge), use inf
    soft_min_closeness = -tau * log_sum_exp
    soft_min_closeness = np.where(np.isfinite(soft_min_closeness), soft_min_closeness, np.inf)
    
    # Normalize edge usage
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_used > 0:
        norm_used = edge_n_used / max_used
    else:
        norm_used = np.zeros_like(edge_n_used)
    
    # Scale penalty to instance: use median of non-zero off-diagonal distances
    off_diag_mask = ~np.eye(n, dtype=bool)
    non_zero_dist = D[off_diag_mask]
    non_zero_dist = non_zero_dist[non_zero_dist > 0]
    if len(non_zero_dist) > 0:
        scale = np.median(non_zero_dist)
    else:
        scale = 0.2
    
    # beta: decay rate relative to instance scale
    # At closeness = scale, force field = exp(-1) ≈ 0.37
    # At closeness = 0, force field = 1.0
    # At closeness = 2*scale, force field = exp(-2) ≈ 0.14
    beta = 1.0 / scale if scale > 0 else 6.0
    
    # Force field: decays with soft-min distance to nearest tour edge
    force_field = np.exp(-beta * soft_min_closeness)
    
    # Usage-based modulation: use sqrt to amplify less-used edges
    usage_mod = np.power(norm_used, 0.5)
    
    # Additive penalty: scale by median edge length for more uniform disruption
    # This breaks 2-opt convergence to previously visited tours more effectively
    gamma = 0.6 * scale
    
    # Additive penalty: D + gamma * force_field * (0.3 + 0.7 * usage_mod)
    penalty = gamma * force_field * (0.3 + 0.7 * usage_mod)
    
    result = D + penalty
    
    # Ensure symmetry and finite values
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 0.0)
    
    # Replace any non-finite values
    result = np.nan_to_num(result, nan=0.0, posinf=D.max() if D.max() > 0 else 1.0, neginf=0.0)
    
    # Safety: ensure all values are positive for off-diagonal
    off_diag = ~np.eye(n, dtype=bool)
    result[off_diag] = np.maximum(result[off_diag], 1e-10)
    
    return result
# EVOLVE-BLOCK-END
