import numpy as np
from scipy.optimize import minimize
import time
import random

N = 50


def _compute_score(centers):
    """Compute the packing score (maximum radius)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    dist_to_center = np.linalg.norm(centers, axis=1)
    if np.max(dist_to_center) >= 1.0:
        return 0.0
    min_d = np.inf
    for i in range(N):
        d = np.linalg.norm(centers - centers[i], axis=1)
        if i > 0:
            min_d = min(min_d, np.min(d[:i]))
        else:
            if N > 1:
                min_d = min(min_d, np.min(d[1:]))
    return min(min_d / 2.0, 1.0 - np.max(dist_to_center))


def _build_points_from_params(params):
    """Convert ring parameters to center positions.
    params: [r1, r2, r3, r4, a1, a2, a3, a4]
    Ring sizes: 1 (center), 6, 12, 12, 19
    """
    ring_sizes = [1, 6, 12, 12, 19]
    centers = np.zeros((N, 2))
    centers[0] = [0.0, 0.0]
    idx = 1
    for i in range(4):
        r = params[i]
        offset = params[4 + i]
        n = ring_sizes[i + 1]
        thetas = np.linspace(0, 2 * np.pi, n, endpoint=False) + offset
        centers[idx:idx + n, 0] = r * np.cos(thetas)
        centers[idx:idx + n, 1] = r * np.sin(thetas)
        idx += n
    return centers


def _parametric_objective(params, beta=100.0):
    """
    Smooth surrogate objective for ring parameters.
    params: [r1, r2, r3, r4, a1, a2, a3, a4]
    Returns negative smooth radius estimate.
    """
    radii = np.abs(params[:4])
    radii = np.clip(radii, 0.01, 0.98)
    angles = params[4:]
    centers = _build_points_from_params(np.concatenate([radii, angles]))
    
    dist_to_center = np.linalg.norm(centers, axis=1)
    if np.max(dist_to_center) >= 1.0:
        return 10.0 + np.max(dist_to_center)
    
    # Compute all pairwise distances
    n = len(centers)
    pair_dists = []
    for i in range(n):
        d = np.linalg.norm(centers - centers[i], axis=1)
        if i > 0:
            pair_dists.extend(d[:i].tolist())
        else:
            pair_dists.extend(d[1:].tolist())
    pair_dists = np.array(pair_dists)
    
    # Boundary distances
    boundary_dists = 1.0 - dist_to_center
    
    # Soft-min over all constraints
    all_constraints = np.concatenate([pair_dists / 2.0, boundary_dists])
    
    m = np.min(all_constraints)
    if not np.isfinite(m):
        return 0.0
    shifted = all_constraints - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = m - log_sum / beta
    return -soft_min_val


def _soft_min_surrogate(values, beta):
    """Compute soft-min of values using log-sum-exp trick."""
    m = np.min(values)
    if not np.isfinite(m):
        return 0.0
    shifted = values - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return m - log_sum / beta


def _smooth_objective(centers_flat, beta=100.0):
    """
    Smooth surrogate objective for full 100-dimensional coordinate vector.
    centers_flat: array of shape (100,)
    Returns negative smooth radius estimate.
    """
    centers = centers_flat.reshape((N, 2))
    
    dist_to_center = np.linalg.norm(centers, axis=1)
    if np.max(dist_to_center) >= 1.0:
        return 10.0 + np.max(dist_to_center)
    
    # Compute all pairwise distances
    n = len(centers)
    pair_dists = []
    for i in range(n):
        d = np.linalg.norm(centers - centers[i], axis=1)
        if i > 0:
            pair_dists.extend(d[:i].tolist())
        else:
            pair_dists.extend(d[1:].tolist())
    pair_dists = np.array(pair_dists)
    
    # Boundary distances
    boundary_dists = 1.0 - dist_to_center
    
    # Soft-min over all constraints
    all_constraints = np.concatenate([pair_dists / 2.0, boundary_dists])
    
    soft_min_val = _soft_min_surrogate(all_constraints, beta)
    return -soft_min_val


def _exact_objective(centers_flat):
    """Exact objective: negative of the packing radius."""
    centers = centers_flat.reshape((N, 2))
    score = _compute_score(centers)
    return -score


def _polish_with_smooth(centers, beta=200.0, maxiter=500):
    """Polish centers using smooth surrogate with L-BFGS-B."""
    try:
        res = minimize(_smooth_objective, centers.flatten(), args=(beta,), method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.fun < 0:
            candidate = res.x.reshape((N, 2))
            if np.all(np.isfinite(candidate)):
                return candidate
    except Exception:
        pass
    return centers


def _exact_polish(centers, max_iters=100, tol=1e-12):
    """
    Exact coordinate descent polish: iteratively move each circle to maximize
    the minimum constraint (pairwise distance or boundary distance).
    Returns (polished_centers, score).
    """
    polished = centers.copy()
    best_score = _compute_score(polished)
    
    for iteration in range(max_iters):
        improved = False
        for i in range(N):
            # Current position
            x, y = polished[i]
            
            # Find the current minimum constraint involving circle i
            # Boundary constraint
            dist_to_origin = np.sqrt(x*x + y*y)
            boundary_margin = 1.0 - dist_to_origin
            
            # Pairwise constraints
            min_pair_dist = np.inf
            for j in range(N):
                if j == i:
                    continue
                dx = x - polished[j, 0]
                dy = y - polished[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                min_pair_dist = min(min_pair_dist, d)
            
            current_min = min(boundary_margin, min_pair_dist / 2.0)
            
            # Try small moves in 8 directions
            step = max(0.001, 0.01 * current_min)
            best_pos = (x, y)
            best_local_min = current_min
            
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    ndist = np.sqrt(nx*nx + ny*ny)
                    if ndist >= 1.0:
                        continue
                    nboundary = 1.0 - ndist
                    nmin_pair = np.inf
                    for j in range(N):
                        if j == i:
                            continue
                        ddx = nx - polished[j, 0]
                        ddy = ny - polished[j, 1]
                        d = np.sqrt(ddx*ddx + ddy*ddy)
                        nmin_pair = min(nmin_pair, d)
                    nlocal = min(nboundary, nmin_pair / 2.0)
                    if nlocal > best_local_min + 1e-15:
                        best_local_min = nlocal
                        best_pos = (nx, ny)
            
            if best_pos != (x, y):
                polished[i, 0] = best_pos[0]
                polished[i, 1] = best_pos[1]
                improved = True
        
        if not improved:
            break
        
        new_score = _compute_score(polished)
        if new_score > best_score:
            best_score = new_score
        if abs(new_score - best_score) < tol:
            break
    
    return polished, best_score


def _multi_start_local_search(n_starts=8, time_budget=60.0):
    """
    Multi-start local search: try multiple initial configurations and polish each.
    Returns the best centers and score found.
    """
    start_time = time.time()
    best_centers = None
    best_score = 0.0
    
    # Generate diverse initial configurations
    inits = []
    
    # Base ring configuration
    base_radii = np.array([0.252, 0.454, 0.655, 0.855])
    base_angles = np.array([0.0, np.pi / 12, np.pi / 6, 0.0])
    base_params = np.concatenate([base_radii, base_angles])
    inits.append(_build_points_from_params(base_params))
    
    # Perturbed ring configurations with varying perturbation scales
    rng = np.random.RandomState(42)
    perturbation_scales = [0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 0.30]
    for k in range(n_starts - 1):
        params = base_params.copy()
        scale = perturbation_scales[k % len(perturbation_scales)]
        params[:4] += rng.randn(4) * scale
        params[:4] = np.clip(np.abs(params[:4]), 0.1, 0.95)
        params[4:] += rng.randn(4) * scale * 6.0
        inits.append(_build_points_from_params(params))
    
    for init in inits:
        if time.time() - start_time > time_budget:
            break
        
        # Smooth polish with two beta values
        polished = _polish_with_smooth(init, beta=80.0, maxiter=150)
        score = _compute_score(polished)
        
        if score > best_score:
            best_score = score
            best_centers = polished.copy()
        
        # Second polish with higher beta if time allows
        if time.time() - start_time < time_budget - 5.0:
            polished2 = _polish_with_smooth(polished, beta=200.0, maxiter=100)
            score2 = _compute_score(polished2)
            if score2 > best_score:
                best_score = score2
                best_centers = polished2.copy()
    
    return best_centers, best_score


def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Phase 1: Multi-start with ring parameter optimization
    best_centers, best_score = _multi_start_local_search(n_starts=8, time_budget=35.0)
    
    if best_centers is None:
        # Fallback to basic ring packing
        base_radii = np.array([0.252, 0.454, 0.655, 0.855])
        base_angles = np.array([0.0, np.pi / 12, np.pi / 6, 0.0])
        best_centers = _build_points_from_params(np.concatenate([base_radii, base_angles]))
        best_score = _compute_score(best_centers)
    
    # Phase 2: Smooth polish with increasing beta
    if time.time() - start_time < time_limit - 25.0:
        for beta in [100, 200, 500]:
            if time.time() - start_time > time_limit - 20.0:
                break
            polished = _polish_with_smooth(best_centers, beta=beta, maxiter=250)
            score = _compute_score(polished)
            if score > best_score:
                best_score = score
                best_centers = polished
    
    # Phase 3: Exact coordinate descent polish (multiple rounds)
    if time.time() - start_time < time_limit - 10.0:
        try:
            polished, polished_score = _exact_polish(best_centers, max_iters=50, tol=1e-12)
            if polished_score > best_score:
                best_score = polished_score
                best_centers = polished
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish if time remains
    if time.time() - start_time < time_limit - 3.0:
        try:
            polished, polished_score = _exact_polish(best_centers, max_iters=30, tol=1e-14)
            if polished_score > best_score:
                best_score = polished_score
                best_centers = polished
        except Exception:
            pass
    
    # Final validation
    if best_centers.shape != (N, 2) or not np.all(np.isfinite(best_centers)):
        # Fallback: basic ring packing
        base_radii = np.array([0.252, 0.454, 0.655, 0.855])
        base_angles = np.array([0.0, np.pi / 12, np.pi / 6, 0.0])
        best_centers = _build_points_from_params(np.concatenate([base_radii, base_angles]))
    
    return best_centers
