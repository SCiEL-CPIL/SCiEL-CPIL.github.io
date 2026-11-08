import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _hull_area(pts):
    """Compute convex hull area."""
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(pts):
    """Compute all triangle areas formed by triples of points."""
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vector from i to j
            dx_ij = x[j] - x[i]
            dy_ij = y[j] - y[i]
            
            # Vectors from i to k for all k > j
            dx_ik = x[j+1:] - x[i]
            dy_ik = y[j+1:] - y[i]
            
            # Cross product magnitude / 2 gives area
            cross = dx_ij * dy_ik - dy_ij * dx_ik
            areas.extend(0.5 * np.abs(cross))
    return np.array(areas)

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 points."""
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _score(pts):
    """Compute the score: min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-10:
        return 0.0
    min_a = _min_triangle_area(pts)
    return min_a / hull_a

def _neg_hard_score(pts_flat):
    """Negative hard score for minimization (used for final polish)."""
    pts = pts_flat.reshape(-1, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    return -_score(pts)

def _soft_objective(pts_flat, beta):
    """
    Soft objective: hull_area / soft_min(triangle_areas).
    Minimizing this maximizes the ratio of min-area to hull-area in a differentiable way.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    """
    pts = pts_flat.reshape(-1, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    
    hull_a = _hull_area(pts)
    if hull_a < 1e-10:
        return 1e10
        
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 1e10
        
    # Avoid log(0) or overflow
    min_a = np.min(areas)
    if min_a < 1e-12:
        return 1e10
        
    # Compute soft min
    # z = -beta * areas
    # soft_min = -1/beta * log(sum(exp(z)))
    # To be numerically stable: max_z = max(z), sum(exp(z - max_z))
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500: # exp underflow protection, though unlikely with positive areas
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
        
    soft_min_val = -log_sum / beta
    
    if soft_min_val < 1e-12:
        return 1e10
        
    # We want to maximize soft_min_val / hull_a
    # So we minimize hull_a / soft_min_val
    return hull_a / soft_min_val

def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 1e-10:
        return pts / np.sqrt(hull_a)
    return pts

def _initial_regular_polygon(n):
    """Generate points of a regular n-gon normalized to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)

def _initial_mixed(n, rng):
    """
    Generate points with some on the boundary of a regular polygon and some inside.
    This helps break symmetry and potentially find better packing.
    """
    outer_n = 8
    inner_n = n - outer_n
    
    # Outer ring
    angles_out = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    
    # Inner points
    # Place them in a smaller rotated polygon or random within
    angles_in = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    # Randomize radii slightly to avoid perfect symmetry
    radii_in = rng.uniform(0.3, 0.6, inner_n)
    inner = np.column_stack([radii_in * np.cos(angles_in), radii_in * np.sin(angles_in)])
    
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)

def _initial_perturbed_regular(n, rng, noise_scale=0.1):
    """Regular polygon with random perturbation."""
    base = _initial_regular_polygon(n)
    noise = rng.normal(0, noise_scale, base.shape)
    pts = base + noise
    return _normalize_unit_area(pts)

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    best_pts = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    # Define diverse initializations
    inits = []
    inits.append(_initial_regular_polygon(N))
    inits.append(_initial_perturbed_regular(N, rng, 0.05))
    inits.append(_initial_perturbed_regular(N, rng, 0.1))
    inits.append(_initial_perturbed_regular(N, rng, 0.2))
    inits.append(_initial_mixed(N, rng))
    
    # Optimization Strategy:
    # 1. Soft Annealing with L-BFGS-B (Fast, smooth)
    # 2. Hard Polish with Nelder-Mead (Precise, non-smooth)
    
    betas = [10, 50, 200, 1000]
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        current_x = init.flatten()
        current_score = _score(init)
        
        # Stage 1: Soft Annealing
        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Define objective for this beta
            def obj_func(x, b=beta):
                return _soft_objective(x, b)
                
            try:
                # L-BFGS-B is efficient for smooth problems
                # We don't have bounds, but we can rely on the normalization in score/obj
                # Actually, the objective is scale invariant (ratio), so L-BFGS-B might drift.
                # However, the soft min / hull area ratio is well-behaved near the optimum.
                
                # Limit iterations to ensure we don't hang
                res = minimize(obj_func, current_x, method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7,
                                        'maxfun': 200})
                
                if res.x is not None:
                    cand = res.x.reshape(-1, 2)
                    # Evaluate hard score to track progress
                    s = _score(cand)
                    if s > current_score:
                        current_score = s
                        current_x = cand
            except Exception:
                # If L-BFGS fails, try Nelder-Mead on soft obj as fallback? 
                # Or just skip to next beta.
                pass
        
        # Stage 2: Hard Polish with Nelder-Mead
        # The soft min is an approximation. We need to snap to the true minimum triangle constraint.
        if time.time() - start_time < DEADLINE - 2:
            try:
                res = minimize(_neg_hard_score, current_x, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                if res.x is not None:
                    cand = res.x.reshape(-1, 2)
                    s = _score(cand)
                    if s > current_score:
                        current_score = s
                        current_x = cand
            except Exception:
                pass
        
        # Update global best
        if current_score > best_score:
            best_score = current_score
            best_pts = current_x.reshape(-1, 2)

    # Final global polish if time permits
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        try:
            res = minimize(_neg_hard_score, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass

    if best_pts is None:
        best_pts = _initial_regular_polygon(N)

    return best_pts
