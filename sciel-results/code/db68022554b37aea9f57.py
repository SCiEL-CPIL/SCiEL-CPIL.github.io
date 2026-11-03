"""Improved program for packing 50 equal circles in a unit circle."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)

def _soft_min_score(P, beta=50.0):
    """
    Compute a smooth approximation of the packing radius using log-sum-exp soft-min.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    pair_dist_half = np.sqrt(dist_sq + 1e-15) / 2.0
    
    dist_origin = np.linalg.norm(P, axis=1)
    bound_dist = 1.0 - dist_origin
    
    iu = np.triu_indices(N, k=1)
    pair_vals = pair_dist_half[iu]
    
    all_vals = np.concatenate([pair_vals, bound_dist])
    
    min_val = np.min(all_vals)
    exp_terms = np.exp(-beta * (all_vals - min_val))
    log_sum = np.log(np.sum(exp_terms))
    
    r_soft = min_val - log_sum / beta
    
    return r_soft

def _energy(P, r):
    """Energy for simulated annealing: penalize violations of radius r."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    dist = np.sqrt(dist_sq + 1e-15)
    cc_violation = np.maximum(2.0 * r - dist, 0.0)
    
    dist_origin = np.linalg.norm(P, axis=1)
    cb_violation = np.maximum(dist_origin + r - 1.0, 0.0)
    
    return np.sum(cc_violation**2) + np.sum(cb_violation**2)

def _energy_grad(P, r):
    """Gradient of the energy function with respect to positions."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq + 1e-15)
    
    grad = np.zeros_like(P)
    
    # Circle-circle gradient
    overlap_cc = np.maximum(2.0 * r - dist, 0.0)
    mask_cc = overlap_cc > 0
    if np.any(mask_cc):
        inv_dist = np.zeros_like(dist)
        np.divide(1.0, dist, out=inv_dist, where=dist > 0)
        weight = np.zeros((N, N))
        weight[mask_cc] = overlap_cc[mask_cc] * inv_dist[mask_cc]
        sum_w = np.sum(weight, axis=1)
        term1 = P * sum_w[:, np.newaxis]
        term2 = (P.T @ weight).T
        grad += 2.0 * (term1 - term2)
    
    # Boundary gradient
    dist_origin = np.linalg.norm(P, axis=1)
    overlap_cb = np.maximum(dist_origin + r - 1.0, 0.0)
    mask_cb = overlap_cb > 0
    if np.any(mask_cb):
        unit_vec = P[mask_cb] / dist_origin[mask_cb, np.newaxis]
        grad[mask_cb] -= 2.0 * unit_vec * overlap_cb[mask_cb, np.newaxis]
    
    return grad

def _get_ring_initialization():
    """Generate a ring-structured initialization."""
    points = []
    points.append([0.0, 0.0])
    
    r1 = 0.3
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    r2 = 0.55
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi/12
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    r3 = 0.78
    for i in range(18):
        angle = 2 * np.pi * i / 18
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    r4 = 0.92
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi/13
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    return np.array(points)

def _get_hex_initial_points():
    """Generate a distorted hexagonal lattice and select the best N points."""
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

def _ring_to_positions(ring_params):
    """Convert ring parameters to 2D positions.
    ring_params: array of shape (n_rings * 2,) where each pair is (radius, angle_offset)
    Ring structure: [center(1), ring1(6), ring2(12), ring3(18), ring4(13)]
    """
    # Ring sizes
    ring_sizes = [1, 6, 12, 18, 13]
    
    points = []
    param_idx = 0
    
    for ring_idx, size in enumerate(ring_sizes):
        if size == 1:
            # Center point
            points.append([0.0, 0.0])
        else:
            radius = ring_params[param_idx]
            angle_offset = ring_params[param_idx + 1]
            param_idx += 2
            
            for i in range(size):
                angle = 2 * np.pi * i / size + angle_offset
                points.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    return np.array(points)

def _optimize_ring_params(ring_params, max_iter=200):
    """Optimize ring parameters using L-BFGS-B on smooth surrogate."""
    def objective(params):
        P = _ring_to_positions(params)
        # Penalize points outside unit circle
        norms = np.linalg.norm(P, axis=1)
        outside_penalty = np.sum(np.maximum(norms - 1.0, 0.0)**2) * 100.0
        r_soft = _soft_min_score(P, beta=50.0)
        return -r_soft + outside_penalty
    
    try:
        result = minimize(objective, ring_params, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
        P_opt = _ring_to_positions(result.x)
        score_opt = _compute_score(P_opt)
        return result.x, P_opt, score_opt
    except:
        return ring_params, _ring_to_positions(ring_params), _compute_score(_ring_to_positions(ring_params))

def _global_search_ring_params(start_time, deadline):
    """Global search over ring parameter space."""
    # Initial ring params: [r1, a1, r2, a2, r3, a3, r4, a4]
    best_params = np.array([0.3, 0.0, 0.55, np.pi/12, 0.78, 0.0, 0.92, np.pi/13])
    best_score = -np.inf
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb initial params
        noise_scale = 0.05 + 0.05 * restart
        params = best_params + np.random.normal(0, noise_scale, len(best_params))
        # Clamp radii to reasonable range
        for i in range(0, len(params), 2):
            params[i] = np.clip(params[i], 0.1, 0.95)
        
        opt_params, P_opt, score_opt = _optimize_ring_params(params, max_iter=150)
        
        if score_opt > best_score:
            best_score = score_opt
            best_params = opt_params
    
    P_best = _ring_to_positions(best_params)
    return P_best, best_score

def _local_refine(P, r, max_iter=300, tol=1e-12, use_smooth=True):
    """Local refinement using smooth surrogate with L-BFGS-B, then exact polish."""
    P_curr = P.copy()
    r_curr = r
    
    # Phase 1: Smooth surrogate optimization with L-BFGS-B
    if use_smooth:
        def smooth_objective(P_flat):
            P_test = P_flat.reshape(N, 2)
            r_soft = _soft_min_score(P_test, beta=50.0)
            norms = np.linalg.norm(P_test, axis=1)
            outside_penalty = np.sum(np.maximum(norms - 1.0, 0.0)**2) * 100.0
            return -r_soft + outside_penalty
        
        P_flat = P_curr.flatten()
        
        try:
            result = minimize(smooth_objective, P_flat, method='L-BFGS-B', 
                             options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol,
                                     'maxls': 50})
            P_opt = result.x.reshape(N, 2)
            score_opt = _compute_score(P_opt)
            if score_opt > r_curr:
                P_curr = P_opt
                r_curr = score_opt
        except:
            pass
    
    # Phase 2: Exact objective polish with gradient descent
    for it in range(max_iter // 2):
        grad = _energy_grad(P_curr, r_curr)
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-15:
            break
        
        best_step = 0
        best_score = r_curr
        for factor in [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
            step = factor * norm_grad / (N * N + 1)
            P_new = P_curr + step * grad
            new_score = _compute_score(P_new)
            if new_score > best_score:
                best_score = new_score
                best_step = step
        
        if best_step > 0:
            P_curr = P_curr + best_step * grad
            r_curr = best_score
        else:
            break
        
        if r_curr - best_score < tol:
            break
    
    return P_curr, r_curr

def _simulated_annealing(P_init, r_init, start_time, deadline):
    """Simulated annealing with adaptive perturbation and periodic local refinement."""
    P_curr = P_init.copy()
    r_curr = r_init
    score_best = r_curr
    P_best = P_init.copy()
    
    T_start = 0.05
    T_end = 0.001
    max_iter = 5000
    no_improve_count = 0
    max_no_improve = 500
    
    for it in range(max_iter):
        if time.time() > deadline - 15.0:
            break
        
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        perturb_scale = 0.02 + 0.03 * (1 + no_improve_count / max_no_improve)
        n_perturb = np.random.randint(1, 8)
        idx = np.random.choice(N, n_perturb, replace=False)
        P_trial = P_curr.copy()
        P_trial[idx] += np.random.normal(0, perturb_scale, (n_perturb, 2))
        
        norms = np.linalg.norm(P_trial, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            P_trial[mask] *= 0.98 / norms[mask, np.newaxis]
        
        E_curr = _energy(P_curr, r_curr)
        E_trial = _energy(P_trial, r_curr)
        dE = E_trial - E_curr
        
        if dE < 0 or np.random.random() < np.exp(-dE / T):
            P_curr = P_trial
            new_score = _compute_score(P_curr)
            if new_score > score_best:
                r_curr = new_score
                score_best = new_score
                P_best = P_curr.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if it % 150 == 0 and it > 0:
            P_ref, score_ref = _local_refine(P_curr, r_curr, max_iter=80, use_smooth=True)
            if score_ref > score_best:
                P_curr = P_ref
                r_curr = score_ref
                score_best = score_ref
                P_best = P_ref.copy()
                no_improve_count = 0
    
    return P_best, score_best

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    # Try multiple initializations and pick the best
    initializations = []
    
    P_hex = _get_hex_initial_points()
    initializations.append(P_hex)
    
    P_ring = _get_ring_initialization()
    initializations.append(P_ring)
    
    for _ in range(3):
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.random.uniform(0, 0.9, N)
        P_rand = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(P_rand)
    
    P_best = initializations[0]
    score_best = _compute_score(P_best)
    for P_init in initializations[1:]:
        s = _compute_score(P_init)
        if s > score_best:
            P_best = P_init
            score_best = s
    
    # Initial local refinement on best init
    P_ref, score_ref = _local_refine(P_best, score_best, max_iter=200, use_smooth=True)
    if score_ref > score_best:
        P_best = P_ref
        score_best = score_ref
    
    # Phase 1: Ring-parametric global search
    if time.time() < deadline - 50.0:
        P_ring_opt, score_ring = _global_search_ring_params(start_time, deadline)
        if score_ring > score_best:
            P_best = P_ring_opt
            score_best = score_ring
            # Refine the ring-optimized solution
            P_ref, score_ref = _local_refine(P_best, score_best, max_iter=200, use_smooth=True)
            if score_ref > score_best:
                P_best = P_ref
                score_best = score_ref
    
    # Phase 2: Simulated annealing with periodic local refinement
    if time.time() < deadline - 30.0:
        P_sa, score_sa = _simulated_annealing(P_best, score_best, start_time, deadline)
        if score_sa > score_best:
            P_best = P_sa
            score_best = score_sa
    
    # Phase 3: Multiple local refinement passes from different perturbations
    if time.time() < deadline - 15.0:
        n_restarts = 6
        for restart in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            scale = 0.02 + 0.02 * restart
            P_trial = P_best + np.random.normal(0, scale, (N, 2))
            norms = np.linalg.norm(P_trial, axis=1)
            mask = norms > 0.95
            if np.any(mask):
                P_trial[mask] *= 0.95 / norms[mask, np.newaxis]
            
            P_ref, score_ref = _local_refine(P_trial, _compute_score(P_trial), max_iter=150, use_smooth=True)
            if score_ref > score_best:
                P_best = P_ref
                score_best = score_ref
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 3.0:
        P_polish, score_polish = _local_refine(P_best, score_best, max_iter=500, tol=1e-14, use_smooth=False)
        if score_polish > score_best:
            P_best = P_polish
            score_best = score_polish
    
    # Ensure all points are inside the unit circle
    norms = np.linalg.norm(P_best, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        P_best[mask] *= (1.0 - 1e-9) / norms[mask, np.newaxis]
    
    return P_best
# EVOLVE-BLOCK-END
