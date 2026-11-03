import numpy as np
from scipy.optimize import minimize
import time

N = 50


def _pairwise_min_dist(centers):
    """Compute minimum pairwise distance between centers."""
    n = len(centers)
    if n < 2:
        return np.inf
    min_d = np.inf
    for i in range(n):
        d = np.linalg.norm(centers - centers[i], axis=1)
        if i > 0:
            min_d = min(min_d, np.min(d[:i]))
        else:
            if n > 1:
                min_d = min(min_d, np.min(d[1:]))
    return min_d


def _score(centers):
    """Compute the packing score (maximum radius)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    dist_to_center = np.linalg.norm(centers, axis=1)
    if np.max(dist_to_center) >= 1.0:
        return 0.0
    min_d = _pairwise_min_dist(centers)
    return min(min_d / 2.0, 1.0 - np.max(dist_to_center))


def _ring_params_to_points(radii, angles):
    """Convert ring parameters to center positions."""
    # radii: [r1, r2, r3, r4]
    # angles: [a1, a2, a3, a4]
    ring_sizes = [6, 12, 12, 19]
    centers = np.zeros((N, 2))
    centers[0] = [0.0, 0.0]
    idx = 1
    for i in range(4):
        r = radii[i]
        offset = angles[i]
        n = ring_sizes[i]
        thetas = np.linspace(0, 2 * np.pi, n, endpoint=False) + offset
        centers[idx:idx + n, 0] = r * np.cos(thetas)
        centers[idx:idx + n, 1] = r * np.sin(thetas)
        idx += n
    return centers


def _soft_min(values, beta):
    """Compute soft-min of values using log-sum-exp trick."""
    m = np.min(values)
    if not np.isfinite(m):
        return 0.0
    shifted = values - m
    # For stability, if beta is very large, exp(-beta*shifted) might underflow for large shifted
    # But shifted >= 0, so exp(-beta*shifted) <= 1.
    # log_sum = log(sum(exp(-beta * shifted)))
    # soft_min = m - log_sum / beta
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return m - log_sum / beta


def _smooth_objective_ring(params, beta=100.0):
    """
    Smooth surrogate objective for ring parameters.
    params: [r1, r2, r3, r4, a1, a2, a3, a4]
    Returns negative smooth radius estimate.
    """
    radii = np.abs(params[:4])
    radii = np.clip(radii, 0.01, 0.98)
    angles = params[4:]
    centers = _ring_params_to_points(radii, angles)
    
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
    
    soft_min_val = _soft_min(all_constraints, beta)
    return -soft_min_val


def _smooth_objective_full(centers_flat, beta=100.0):
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
    
    soft_min_val = _soft_min(all_constraints, beta)
    return -soft_min_val


def _ring_initialization():
    """Generate initial ring parameters based on known good packing."""
    # Known packing: 1 center, rings of 6, 12, 12, 19
    # Approximate ring radii for r_circle ≈ 0.1258
    base_radii = np.array([0.252, 0.454, 0.655, 0.855])
    base_angles = np.array([0.0, np.pi / 12, np.pi / 6, 0.0])
    return np.concatenate([base_radii, base_angles])


def _multiple_initializations(n_initials=5):
    """Generate multiple diverse initial conditions."""
    inits = [_ring_initialization()]
    np.random.seed(12345)
    for _ in range(n_initials - 1):
        init = _ring_initialization().copy()
        # Perturb radii
        init[:4] += np.random.randn(4) * 0.05
        init[:4] = np.clip(np.abs(init[:4]), 0.1, 0.95)
        # Perturb angles
        init[4:] += np.random.randn(4) * 0.3
        inits.append(init)
    return inits


def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    best_params = _ring_initialization()
    best_score = _score(_ring_params_to_points(
        np.clip(np.abs(best_params[:4]), 0.01, 0.98), best_params[4:]))
    best_centers = _ring_params_to_points(
        np.clip(np.abs(best_params[:4]), 0.01, 0.98), best_params[4:])
    
    inits = _multiple_initializations(5)
    
    # Phase 1: Optimize ring parameters with increasing beta
    betas = [10, 50, 100, 200]
    current_params = best_params.copy()
    
    for beta in betas:
        if time.time() - start_time > time_limit - 30.0:
            break
            
        for init in inits:
            if time.time() - start_time > time_limit - 30.0:
                break
                
            try:
                bounds = [(0.05, 0.95)] * 4 + [(-np.pi, np.pi)] * 4
                # Use L-BFGS-B for the smooth objective
                res = minimize(_smooth_objective_ring, init, args=(beta,), method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
                
                if res.fun < -best_score:
                    candidate = res.x.copy()
                    candidate[:4] = np.clip(np.abs(candidate[:4]), 0.01, 0.98)
                    candidate_score = _score(_ring_params_to_points(candidate[:4], candidate[4:]))
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_params = candidate
                        best_centers = _ring_params_to_points(candidate[:4], candidate[4:])
                        current_params = candidate
            except Exception:
                pass
    
    # Phase 2: Polish with full coordinates
    # Convert best ring params to full coordinates
    best_centers_flat = best_centers.flatten()
    
    if time.time() - start_time < time_limit - 10.0:
        # Use a high beta for the full coordinate optimization
        beta_full = 200.0
        try:
            # L-BFGS-B on 100 dimensions
            # Bounds: x, y in [-1, 1] is too loose, but centers must be inside unit circle.
            # We rely on the objective penalty for boundary violation.
            res = minimize(_smooth_objective_full, best_centers_flat, args=(beta_full,), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            candidate_centers = res.x.reshape((N, 2))
            
            # Check if candidate is valid and better
            if np.all(np.isfinite(candidate_centers)):
                candidate_score = _score(candidate_centers)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_centers = candidate_centers
        except Exception:
            pass
    
    # Phase 3: Exact local search on full coordinates
    if time.time() - start_time < time_limit - 5.0:
        # Try a few random perturbations and Nelder-Mead on exact objective
        # This is risky due to non-smoothness, but might catch small improvements
        # We'll skip complex N-M on 100D as it's slow. 
        # Instead, we do a simple gradient step on the exact objective using finite differences?
        # No, let's stick to the smooth result which is usually very close.
        
        # One more polish with even higher beta if time permits
        if time.time() - start_time < time_limit - 2.0:
             try:
                 beta_final = 500.0
                 res = minimize(_smooth_objective_full, best_centers.flatten(), args=(beta_final,), method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14})
                 candidate_centers = res.x.reshape((N, 2))
                 if np.all(np.isfinite(candidate_centers)):
                     candidate_score = _score(candidate_centers)
                     if candidate_score > best_score:
                         best_score = candidate_score
                         best_centers = candidate_centers
             except Exception:
                 pass

    # Final validation
    if best_centers.shape != (N, 2) or not np.all(np.isfinite(best_centers)):
        # Fallback: hexagonal-like packing
        best_centers = _fallback_packing()
    
    return best_centers


def _fallback_packing():
    """Simple fallback packing."""
    centers = np.zeros((N, 2))
    centers[0] = [0.0, 0.0]
    idx = 1
    ring_sizes = [6, 12, 12, 19]
    ring_radii = [0.25, 0.45, 0.65, 0.85]
    ring_angles = [0.0, np.pi/12, np.pi/6, 0.0]
    for i in range(4):
        n = ring_sizes[i]
        r = ring_radii[i]
        offset = ring_angles[i]
        thetas = np.linspace(0, 2*np.pi, n, endpoint=False) + offset
        centers[idx:idx+n, 0] = r * np.cos(thetas)
        centers[idx:idx+n, 1] = r * np.sin(thetas)
        idx += n
    return centers
