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


def _smooth_objective(params, beta=30.0):
    """
    Smooth surrogate objective using soft-min.
    params: [r1, r2, r3, r4, a1, a2, a3, a4]
    Returns negative smooth radius estimate.
    """
    radii = np.abs(params[:4])
    radii = np.clip(radii, 0.01, 0.98)
    angles = params[4:]
    params_clean = np.concatenate([radii, angles])
    centers = _ring_params_to_points(params_clean[:4], params_clean[4:])
    
    # Check boundary
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
    
    # Soft-min via log-sum-exp
    m = np.min(all_constraints)
    if not np.isfinite(m):
        return 10.0
    shifted = all_constraints - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min_val = m - log_sum
    
    return -soft_min_val


def _exact_objective(params):
    """Exact (nonsmooth) objective for final polish."""
    radii = np.abs(params[:4])
    radii = np.clip(radii, 0.01, 0.98)
    angles = params[4:]
    centers = _ring_params_to_points(radii, angles)
    
    dist_to_center = np.linalg.norm(centers, axis=1)
    if np.max(dist_to_center) >= 1.0:
        return 10.0
    
    min_d = _pairwise_min_dist(centers)
    r = min(min_d / 2.0, 1.0 - np.max(dist_to_center))
    return -r


def _ring_initialization():
    """Generate initial ring parameters based on known good packing."""
    # Known packing: 1 center, rings of 6, 12, 12, 19
    # Approximate ring radii for r_circle ≈ 0.1258
    base_radii = np.array([0.252, 0.454, 0.655, 0.855])
    base_angles = np.array([0.0, np.pi / 12, np.pi / 6, 0.0])
    return np.concatenate([base_radii, base_angles])


def _multiple_initializations(n_initials=10):
    """Generate multiple diverse initial conditions."""
    inits = [_ring_initialization()]
    np.random.seed(12345)
    for _ in range(n_initials - 1):
        init = _ring_initialization().copy()
        # Perturb radii
        init[:4] += np.random.randn(4) * 0.03
        init[:4] = np.clip(np.abs(init[:4]), 0.1, 0.95)
        # Perturb angles
        init[4:] += np.random.randn(4) * 0.2
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
    
    inits = _multiple_initializations(8)
    
    # Phase 1: Smooth surrogate optimization with multiple methods
    for init in inits:
        if time.time() - start_time > time_limit - 20.0:
            break
        
        # Try L-BFGS-B with smooth objective
        try:
            bounds = [(0.05, 0.95)] * 4 + [(-np.pi, np.pi)] * 4
            res = minimize(_smooth_objective, init, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < -best_score:
                candidate = res.x.copy()
                candidate[:4] = np.clip(np.abs(candidate[:4]), 0.01, 0.98)
                candidate_score = _score(_ring_params_to_points(candidate[:4], candidate[4:]))
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_params = candidate
                    best_centers = _ring_params_to_points(candidate[:4], candidate[4:])
        except Exception:
            pass
        
        if time.time() - start_time > time_limit - 20.0:
            break
        
        # Try Nelder-Mead with smooth objective
        try:
            res = minimize(_smooth_objective, init, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            
            if res.fun < -best_score:
                candidate = res.x.copy()
                candidate[:4] = np.clip(np.abs(candidate[:4]), 0.01, 0.98)
                candidate_score = _score(_ring_params_to_points(candidate[:4], candidate[4:]))
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_params = candidate
                    best_centers = _ring_params_to_points(candidate[:4], candidate[4:])
        except Exception:
            pass
    
    # Phase 2: Exact objective local refinement
    if time.time() - start_time < time_limit - 10.0:
        for _ in range(3):
            try:
                res = minimize(_exact_objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                
                candidate = res.x.copy()
                candidate[:4] = np.clip(np.abs(candidate[:4]), 0.01, 0.98)
                candidate_score = _score(_ring_params_to_points(candidate[:4], candidate[4:]))
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_params = candidate
                    best_centers = _ring_params_to_points(candidate[:4], candidate[4:])
                else:
                    break
            except Exception:
                break
    
    # Phase 3: Binary search radius polish
    if time.time() - start_time < time_limit - 5.0:
        try:
            # We can try to push the radius up by checking if we can increase it
            # This is a simple local search: try small perturbations
            for trial in range(50):
                if time.time() - start_time > time_limit - 2.0:
                    break
                
                # Random perturbation
                perturbation = np.random.randn(8) * 0.001
                perturbation[:4] *= 0.5  # Smaller perturbation for radii
                candidate = best_params + perturbation
                candidate[:4] = np.clip(np.abs(candidate[:4]), 0.01, 0.98)
                
                candidate_score = _score(_ring_params_to_points(candidate[:4], candidate[4:]))
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_params = candidate
                    best_centers = _ring_params_to_points(candidate[:4], candidate[4:])
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
