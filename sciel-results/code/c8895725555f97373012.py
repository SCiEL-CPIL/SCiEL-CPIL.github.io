"""Pack 50 equal circles in the unit disk via ring-based parametric optimization + multi-stage smooth annealing + constraint push + binary search."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 140.0  # safety margin before 150s limit


def _pairwise_distances(P):
    """Compute all pairwise distances efficiently."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    return np.sqrt(dist_sq)


def _score(P):
    """Compute the packing radius: min of half-min-pair-dist and min-container-margin."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    # Container constraint
    container_margin = 1.0 - np.sqrt(np.sum(P ** 2, axis=1))
    min_container = np.min(container_margin)
    
    # Pairwise distances
    dist = _pairwise_distances(P)
    min_pair = np.min(dist) / 2.0
    
    return min(min_container, min_pair)


def _softmin_logsumexp(x, beta=20.0):
    """Soft-min approximation using log-sum-exp."""
    # -logsumexp(-beta*x)/beta approximates min(x)
    # Use numerically stable form
    x_min = np.min(x)
    return x_min - np.log(np.sum(np.exp(-beta * (x - x_min)))) / beta


def _softmax_logsumexp(x, beta=20.0):
    """Soft-max approximation using log-sum-exp."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta


def _smooth_objective(P, r, beta=20.0):
    """
    Smooth energy function using soft-min approximation.
    beta controls smoothness (higher = sharper).
    """
    # Circle-circle overlaps
    dist = _pairwise_distances(P)
    
    # Soft max for circle-circle overlaps: smooth approximation of max(0, 2r - d)
    raw_cc = 2.0 * r - dist
    scale = beta
    soft_cc = (1.0 / scale) * np.logaddexp(0, scale * raw_cc)
    
    # Circle-boundary overlaps
    dist_origin = np.linalg.norm(P, axis=1)
    raw_cb = dist_origin + r - 1.0
    soft_cb = (1.0 / scale) * np.logaddexp(0, scale * raw_cb)
    
    return 0.5 * (np.sum(soft_cc**2) + np.sum(soft_cb**2))


def _smooth_grad(P, r, beta=20.0):
    """
    Gradient of the smooth energy function.
    """
    grad = np.zeros_like(P)
    scale = beta
    
    # Pairwise
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = _pairwise_distances(P)
    
    raw_cc = 2.0 * r - dist
    sig_cc = 1.0 / (1.0 + np.exp(-scale * raw_cc))
    sig_cc = np.clip(sig_cc, 1e-12, 1.0 - 1e-12)
    
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 0)
    
    soft_cc_val = (1.0 / scale) * np.logaddexp(0, scale * raw_cc)
    
    W = soft_cc_val * sig_cc * inv_dist
    np.fill_diagonal(W, 0.0)
    
    sum_W = np.sum(W, axis=1)
    term1 = (P.T @ W).T
    term2 = P * sum_W[:, np.newaxis]
    
    grad += term1 - term2
    
    # Boundary
    dist_origin = np.linalg.norm(P, axis=1)
    raw_cb = dist_origin + r - 1.0
    sig_cb = 1.0 / (1.0 + np.exp(-scale * raw_cb))
    sig_cb = np.clip(sig_cb, 1e-12, 1.0 - 1e-12)
    
    soft_cb_val = (1.0 / scale) * np.logaddexp(0, scale * raw_cb)
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    
    grad += (soft_cb_val * sig_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad


def _ring_to_points(params, ring_counts):
    """
    Generate points based on concentric rings.
    params: [R1, R2, ..., Rk, offset1, offset2, ..., offsetk]
    ring_counts: list of number of points in each ring.
    """
    P = np.zeros((N, 2))
    k = len(ring_counts)
    
    radii = params[:k]
    offsets = params[k:2*k]
    
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
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


def _anneal_points(P_init, r_target, maxiter=500, beta_start=10.0, beta_end=5000.0, deadline=None):
    """
    Multi-stage smooth annealing: start with low beta, gradually increase.
    Returns (P_opt, min_energy).
    """
    n_stages = 8
    betas = np.logspace(np.log10(beta_start), np.log10(beta_end), n_stages)
    
    P_current = P_init.copy()
    
    for stage, beta in enumerate(betas):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        def objective(P_flat):
            P = P_flat.reshape(N, 2)
            return _smooth_objective(P, r_target, beta=beta)
        
        def grad(P_flat):
            P = P_flat.reshape(N, 2)
            return _smooth_grad(P, r_target, beta=beta).flatten()
            
        try:
            res = minimize(objective, P_current.flatten(), jac=grad, method='L-BFGS-B', 
                           options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50})
            P_current = res.x.reshape(N, 2)
        except Exception:
            pass
    
    min_energy = _smooth_objective(P_current, r_target, beta=beta_end)
    return P_current, min_energy


def _constraint_push(P, r, n_iters=100, deadline=None):
    """
    Constraint push: iteratively push points that violate constraints.
    """
    P = P.copy()
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Check container constraint
        dist_origin = np.linalg.norm(P, axis=1)
        for i in range(N):
            if dist_origin[i] + r > 1.0:
                # Push back
                scale = (1.0 - r) / dist_origin[i]
                P[i] *= scale
        
        # Check pairwise constraints
        dist = _pairwise_distances(P)
        for i in range(N):
            for j in range(i+1, N):
                if dist[i, j] < 2.0 * r:
                    # Push apart
                    mid = (P[i] + P[j]) / 2.0
                    diff = P[i] - P[j]
                    d = np.linalg.norm(diff)
                    if d < 1e-10:
                        continue
                    diff_norm = diff / d
                    push = (2.0 * r - dist[i, j]) / 2.0
                    P[i] += push * diff_norm
                    P[j] -= push * diff_norm
                    
                    # Re-check container
                    for idx in [i, j]:
                        dist_o = np.linalg.norm(P[idx])
                        if dist_o + r > 1.0:
                            scale = (1.0 - r) / dist_o
                            P[idx] *= scale
    
    return P


def _de_ring_search(ring_counts, seed=42, maxiter=80, deadline=None):
    """
    Optimize ring radii and offsets using differential evolution.
    """
    k = len(ring_counts)
    # Bounds: radii in [0, 0.95], offsets in [0, 2*pi]
    bounds = [(0.0, 0.95)] * k + [(0.0, 2 * np.pi)] * k
    
    def neg_score(params):
        if deadline is not None and time.time() > deadline:
            return 0.0  # Infeasible
        P = _ring_to_points(params, ring_counts)
        s = _score(P)
        return -s
        
    try:
        res = differential_evolution(neg_score, bounds, maxiter=maxiter, popsize=20, 
                                     tol=1e-8, seed=seed, polish=False, 
                                     workers=1, init='latinhypercube')
        
        best_params = res.x
        best_P = _ring_to_points(best_params, ring_counts)
        return best_P, -res.fun
        
    except Exception:
        # Fallback: try a few random configurations
        rng = np.random.default_rng(seed)
        best_P = None
        best_score = -np.inf
        for _ in range(30):
            params = np.array([rng.uniform(0.1, 0.9) for _ in range(k)] + 
                             [rng.uniform(0, 2*np.pi) for _ in range(k)])
            P = _ring_to_points(params, ring_counts)
            s = _score(P)
            if s > best_score:
                best_score = s
                best_P = P
        if best_P is None:
            best_P = _get_hex_initial_points()
            best_score = _score(best_P)
        return best_P, best_score


def _get_hex_initial_points():
    """Fallback initialization using hexagonal lattice."""
    spacing = 0.22
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0 
        mask = radius <= 1.0 - margin
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.95
        if spacing < 0.1:
            break
    
    # Last resort: grid
    points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * 0.2
            y = j * 0.2
            if x*x + y*y <= 0.85:
                points.append([x, y])
    points = np.array(points[:N])
    return points


def _local_search(pts, n_iters=3000, seed=12345, deadline=None):
    """
    Local search: perturb one point at a time, accept if score improves.
    """
    rng = np.random.default_rng(seed)
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    step_size = 0.01
    temperature = 0.003
    no_improve_count = 0
    max_no_improve = 200
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        old_pos = best_pts[idx].copy()
        new_pos = old_pos + step_size * rng.normal(size=2)
        
        # Keep inside container
        new_r = np.sqrt(np.sum(new_pos ** 2))
        if new_r > 0.995:
            new_pos *= 0.995 / new_r
        
        test_pts = best_pts.copy()
        test_pts[idx] = new_pos
        new_score = _score(test_pts)
        
        delta = new_score - best_score
        if delta > 0 or rng.random() < np.exp(delta / temperature):
            best_pts[idx] = new_pos
            if new_score > best_score:
                best_score = new_score
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            temperature = max(temperature * 2.0, 0.008)
            step_size = min(step_size * 1.5, 0.04)
            no_improve_count = 0
        
        temperature *= 0.99995
        step_size = max(0.001, step_size * 0.99995)
    
    return best_pts, best_score


def _binary_search_radius(P_init, r_low, r_high, max_iterations=15, deadline=None):
    """
    Binary search on radius using smooth optimization to find best feasible packing.
    Returns (best_P, best_score, best_r).
    """
    best_P = P_init.copy()
    best_score = _score(P_init)
    best_r = r_low
    
    for _ in range(max_iterations):
        if deadline is not None and time.time() > deadline - 15.0:
            break
        if r_high - r_low < 1e-10:
            break
            
        r_mid = (r_low + r_high) / 2.0
        P_init_cur = best_P.copy()
        
        # Try to optimize for r_mid with multi-stage annealing
        P_opt, E = _anneal_points(P_init_cur, r_mid, maxiter=300, beta_start=10.0, beta_end=5000.0, deadline=deadline)
        
        # Apply constraint push
        P_opt = _constraint_push(P_opt, r_mid, n_iters=50, deadline=deadline)
        
        score_opt = _score(P_opt)
        
        # Check if feasible: energy near zero and score at least r_mid
        if E < 1e-6 and score_opt >= r_mid * 0.9999:
            if score_opt > best_score:
                best_P = P_opt
                best_score = score_opt
                best_r = r_mid
            r_low = r_mid
        else:
            r_high = r_mid
    
    return best_P, best_score, best_r


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # 5s buffer before 120s limit
    
    # Define ring structures to try
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 12, 17, 14],
        [1, 6, 11, 13, 19],
    ]
    
    # Start with hexagonal initialization
    P_best = _get_hex_initial_points()
    score_best = _score(P_best)
    
    # Try each ring structure with parametric optimization
    for ring_counts in ring_structures:
        if time.time() > deadline - 45.0:
            break
        
        # 1. Global parametric search for ring radii and offsets
        P_candidate, score_candidate = _de_ring_search(
            ring_counts, seed=42, maxiter=70, deadline=deadline - 30.0)
        
        if score_candidate > score_best:
            P_best = P_candidate
            score_best = score_candidate
        
        # 2. Binary search on radius to push further
        if time.time() > deadline - 30.0:
            break
            
        r_low = max(score_best * 0.998, 0.12)
        r_high = 0.148
        
        P_opt, score_opt, _ = _binary_search_radius(
            P_candidate, r_low, r_high, max_iterations=12, deadline=deadline)
        
        if score_opt > score_best:
            P_best = P_opt
            score_best = score_opt
    
    # Final local search refinement - use remaining time
    if time.time() < deadline - 15.0:
        remaining_time = deadline - 15.0 - time.time()
        remaining_iters = min(5000, int(remaining_time * 800))
        remaining_iters = max(1000, remaining_iters)
        P_best, score_best = _local_search(P_best, n_iters=remaining_iters, seed=99999, deadline=deadline - 15.0)
    
    # Final multi-stage polish with progressively higher betas
    if time.time() < deadline - 8.0:
        for beta_pass in [20.0, 100.0, 500.0, 2000.0]:
            if time.time() > deadline - 3.0:
                break
            r_target = score_best * 1.00001
            P_opt, E = _anneal_points(P_best, r_target, maxiter=400, beta_start=10.0, beta_end=beta_pass, deadline=deadline)
            P_opt = _constraint_push(P_opt, r_target, n_iters=30, deadline=deadline)
            polished_score = _score(P_opt)
            if polished_score > score_best:
                P_best = P_opt
                score_best = polished_score
        
        # Ultra-tight polish
        if time.time() < deadline - 2.0:
            r_target = score_best * 1.000005
            P_opt, E = _anneal_points(P_best, r_target, maxiter=500, beta_start=100.0, beta_end=10000.0, deadline=deadline)
            P_opt = _constraint_push(P_opt, r_target, n_iters=50, deadline=deadline)
            polished_score = _score(P_opt)
            if polished_score > score_best:
                P_best = P_opt
                score_best = polished_score
    
    # Validate
    radii = np.sqrt(np.sum(P_best ** 2, axis=1))
    if np.any(~np.isfinite(P_best)) or np.any(radii >= 1.0):
        P_best = _get_hex_initial_points()
    
    return P_best
