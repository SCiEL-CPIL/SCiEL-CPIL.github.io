"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    
    # Distance to boundary
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    # Pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)

def _softmin_logsumexp(values, beta):
    """
    Smooth approximation to min(values) using log-sum-exp.
    For large beta, this approaches the true minimum.
    """
    # Clip to avoid overflow
    values_clipped = np.clip(values, -1e6, 1e6)
    m = np.min(values_clipped)
    if m < -1e5:
        return m
    shifted = values_clipped - m
    # Use log-sum-exp for numerical stability
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return m - np.log(np.sum(np.exp(-beta * shifted))) / beta

def _smooth_objective(P, r_target, beta):
    """
    Smooth surrogate objective: soft-min of all constraints.
    We want to maximize the minimum of:
      - pairwise distances / 2
      - (1 - |c_i|)
    This is equivalent to minimizing the soft-min of the negative constraints,
    or equivalently, we can define a smooth objective that penalizes when
    any constraint is below r_target.
    
    Here we use: for each constraint c_i, penalty = max(0, r_target - c_i)^2
    but smoothed via soft-min.
    
    Actually, let's use a different approach:
    Objective = soft_min of all (constraint_values) 
    where constraint_values = [pairwise_dists/2, boundary_dists]
    We want to maximize this soft_min.
    So we minimize -soft_min(constraint_values).
    """
    # Pairwise distances / 2
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_vals = dist / 2.0
    
    # Boundary distances
    dist_origin = np.linalg.norm(P, axis=1)
    bound_vals = 1.0 - dist_origin
    
    # Collect all constraint values (excluding inf on diagonal)
    # For pairwise, exclude diagonal
    pair_vals_flat = pair_vals[np.triu_indices(N, k=1)]
    all_vals = np.concatenate([pair_vals_flat, bound_vals])
    
    # Soft-min of all values
    sm = _softmin_logsumexp(all_vals, beta)
    
    return -sm  # We minimize the negative

def _smooth_objective_grad(P, r_target, beta):
    """
    Gradient of the smooth objective with respect to P.
    """
    # Pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_vals = dist / 2.0
    
    # Boundary distances
    dist_origin = np.linalg.norm(P, axis=1)
    bound_vals = 1.0 - dist_origin
    
    # Collect all constraint values
    pair_vals_flat = pair_vals[np.triu_indices(N, k=1)]
    all_vals = np.concatenate([pair_vals_flat, bound_vals])
    
    # Soft-min weights: w_i = exp(-beta * (v_i - min_v)) / sum(exp(-beta * (v_j - min_v)))
    # This gives weights that sum to 1, with the minimum value getting weight ~1
    m = np.min(all_vals)
    shifted = all_vals - m
    exp_neg = np.exp(-beta * shifted)
    weights = exp_neg / np.sum(exp_neg)
    
    grad = np.zeros_like(P)
    
    # Pairwise contribution
    # For each pair (i,j) with i<j, the value is dist(i,j)/2
    # d(dist/2)/dP_i = -(P_j - P_i) / (2 * dist)  ... wait
    # dist = |P_i - P_j|
    # d(dist)/dP_i = (P_i - P_j) / dist
    # d(dist/2)/dP_i = (P_i - P_j) / (2 * dist)
    
    w_pair = weights[:N*(N-1)//2]
    
    # Build the pairwise weight matrix
    W_pair = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            W_pair[i, j] = w_pair[idx]
            W_pair[j, i] = w_pair[idx]
            idx += 1
    
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 0)
    
    # Gradient from pairwise: sum_j w_{ij} * (P_i - P_j) / (2 * dist_{ij})
    # = 0.5 * sum_j W_pair[i,j] * (P_i - P_j) / dist_{ij}
    # = 0.5 * (sum_j W_pair[i,j] * P_i / dist_{ij} - sum_j W_pair[i,j] * P_j / dist_{ij})
    # = 0.5 * (P_i * sum_j W_pair[i,j]/dist_{ij} - sum_j W_pair[i,j]*P_j/dist_{ij})
    
    W_over_dist = W_pair * inv_dist
    sum_Wod = np.sum(W_over_dist, axis=1)
    term1 = P * sum_Wod[:, np.newaxis]
    term2 = (P.T @ W_over_dist).T
    
    grad += 0.5 * (term1 - term2)
    
    # Boundary contribution
    w_bound = weights[N*(N-1)//2:]
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    
    # d(bound_val_i)/dP_i = -P_i / |P_i|
    # So gradient contribution = -w_bound_i * P_i / |P_i|
    # But we're minimizing -soft_min, so we add the gradient of -soft_min
    # d(-soft_min)/dP_i = -sum_j w_j * d(v_j)/dP_i
    # For boundary: d(v_i)/dP_i = -P_i/|P_i|
    # So contribution = -w_bound_i * (-P_i/|P_i|) = w_bound_i * P_i/|P_i|
    
    grad += (w_bound * inv_dist_origin)[:, np.newaxis] * P
    
    return grad

def _constraint_push(P, r_target, n_iters=100, step_size=0.01):
    """
    Simple constraint push: iteratively push points that violate constraints.
    """
    P = P.copy()
    for it in range(n_iters):
        # Check pairwise
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # For each pair that's too close, push apart
        for i in range(N):
            for j in range(i+1, N):
                d = dist[i, j]
                if d < 2 * r_target and d > 1e-10:
                    direction = (P[i] - P[j]) / d
                    push = 0.5 * (2 * r_target - d) * step_size
                    P[i] += direction * push
                    P[j] -= direction * push
        
        # Check boundary
        dist_origin = np.linalg.norm(P, axis=1)
        for i in range(N):
            if dist_origin[i] + r_target > 1.0:
                if dist_origin[i] > 1e-10:
                    direction = P[i] / dist_origin[i]
                    push = (dist_origin[i] + r_target - 1.0) * step_size
                    P[i] -= direction * push
                else:
                    # At origin, push in random direction
                    P[i] += np.random.randn(2) * step_size * 0.1
    
    return P

def _get_ring_initialization(params, ring_counts):
    """
    Generate points based on concentric rings.
    params: [R1, R2, ..., Rk, offset1, offset2, ..., offsetk]
    ring_counts: list of number of points in each ring.
    """
    P = np.zeros((N, 2))
    k = len(ring_counts)
    
    # Radii are the first k params
    radii = params[:k]
    # Offsets are the next k params
    offsets = params[k:2*k]
    
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            # Center point
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = radii[i]
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count+n_pts, 0] = R * np.cos(angles)
            P[count:count+n_pts, 1] = R * np.sin(angles)
            count += n_pts
            
    return P

def _optimize_smooth(P_init, beta, time_limit=5.0):
    """
    Optimize positions P to maximize the soft-min of constraints.
    Returns P_opt.
    """
    start_time = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _smooth_objective(P, None, beta)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _smooth_objective_grad(P, None, beta).flatten()
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        P_opt = res.x.reshape(N, 2)
        return P_opt
    except Exception:
        return P_init

def _global_search_ring_params(ring_counts, time_limit=25.0):
    """
    Use differential evolution to find good ring radii and offsets.
    """
    k = len(ring_counts)
    # Bounds for radii: 0 to 1
    # Bounds for offsets: 0 to 2*pi
    bounds = [(0.0, 1.0)] * k + [(0.0, 2 * np.pi)] * k
    
    def neg_score(params):
        P = _get_ring_initialization(params, ring_counts)
        score = _compute_score(P)
        return -score
        
    start_time = time.time()
    
    try:
        res = differential_evolution(neg_score, bounds, maxiter=80, popsize=20, 
                                     tol=1e-6, seed=42, polish=True, 
                                     workers=1,
                                     init='latinhypercube')
        
        best_params = res.x
        best_P = _get_ring_initialization(best_params, ring_counts)
        
    except Exception:
        best_P = _get_hex_initial_points()
        
    return best_P

def _get_hex_initial_points():
    """Fallback initialization using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0 
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer
    
    # Define ring structure
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16]
    ]
    
    P_best = _get_hex_initial_points()
    score_best = _compute_score(P_best)
    
    for ring_counts in ring_structures:
        if time.time() > deadline - 30.0:
            break
            
        # 1. Global search for ring parameters
        P_candidate = _global_search_ring_params(ring_counts, time_limit=20.0)
        score_candidate = _compute_score(P_candidate)
        
        if score_candidate > score_best:
            P_best = P_candidate
            score_best = score_candidate
        
        # 2. Multi-stage smooth optimization with increasing beta
        P_work = P_candidate.copy()
        betas = [10, 50, 200, 1000, 5000]
        
        for beta in betas:
            if time.time() > deadline - 15.0:
                break
            P_work = _optimize_smooth(P_work, beta, time_limit=5.0)
            # Constraint push to fix any violations
            r_est = _compute_score(P_work)
            if r_est > 0:
                P_work = _constraint_push(P_work, r_est, n_iters=50, step_size=0.005)
        
        score_work = _compute_score(P_work)
        if score_work > score_best:
            P_best = P_work
            score_best = score_work
        
        # 3. Final high-precision polish with very high beta
        if time.time() < deadline - 10.0:
            P_final = _optimize_smooth(P_best, 10000, time_limit=5.0)
            r_est = _compute_score(P_final)
            if r_est > 0:
                P_final = _constraint_push(P_final, r_est, n_iters=100, step_size=0.002)
            score_final = _compute_score(P_final)
            if score_final > score_best:
                P_best = P_final
                score_best = score_final
    
    # Final polish on best result
    if time.time() < deadline - 3.0:
        P_polished = _optimize_smooth(P_best, 20000, time_limit=5.0)
        r_est = _compute_score(P_polished)
        if r_est > 0:
            P_polished = _constraint_push(P_polished, r_est, n_iters=200, step_size=0.001)
        score_polished = _compute_score(P_polished)
        if score_polished > score_best:
            P_best = P_polished
            score_best = score_polished
            
    return P_best
# EVOLVE-BLOCK-END
