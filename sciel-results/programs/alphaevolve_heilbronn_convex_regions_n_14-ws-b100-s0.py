"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

DEADLINE = 145.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s


def _soft_obj(pts_flat, beta=100.0):
    """Soft objective: hull_area / soft_min(triangle areas)."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val


def _multi_beta_soft_obj(pts_flat, betas=(50.0, 150.0, 300.0)):
    """Multi-beta soft objective: weighted combination of soft-mins."""
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    total = 0.0
    for beta in betas:
        z = -beta * areas
        max_z = np.max(z)
        if max_z < -500:
            log_sum = max_z
        else:
            log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
        soft_min_val = -log_sum / beta
        if soft_min_val < 1e-12:
            return 1e10
        total += hull_area / soft_min_val
    return total / len(betas)


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_mixed(n):
    """Mix of boundary and interior points for better minimum triangle area."""
    outer_n = 10
    inner_n = n - outer_n
    angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.4 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def _initial_hexagonal(n):
    """Hexagonal lattice inspired initialization."""
    pts = []
    pts.append([0.0, 0.0])
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([np.cos(angle), np.sin(angle)])
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.pi / 7
        r = 1.8
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    
    pts = np.array(pts)
    return _normalize_unit_area(pts)


def _initial_cosine_mod(n):
    """Regular polygon with cosine-modulated radii."""
    angles = 2 * np.pi * np.arange(n) / n
    mod = 1.0 + 0.1 * np.cos(3 * angles)
    pts = np.column_stack([mod * np.cos(angles), mod * np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_grid_based(n):
    """Grid-based initialization with perturbation."""
    rng = np.random.RandomState(123)
    angles = rng.uniform(0, 2*np.pi, n)
    radii = rng.uniform(0.3, 1.0, n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _normalize_unit_area(pts)


def _optimize_with_annealing(init_pts, time_budget):
    """Optimize using beta annealing with L-BFGS-B."""
    start_time = time.time()
    x0 = init_pts.flatten()
    
    best_x = x0.copy()
    best_obj = _soft_obj(x0, beta=50.0)
    
    # Annealing schedule: start with low beta (smooth), increase to high beta (sharp)
    betas = [30, 50, 100, 200, 400, 800]
    
    for beta in betas:
        if time.time() - start_time > time_budget - 5:
            break
        
        def obj_func(x):
            return _soft_obj(x, beta=beta)
        
        try:
            # Estimate how many iterations we can afford
            remaining = time_budget - (time.time() - start_time) - 5
            iters_per_stage = max(20, int(remaining / (len(betas) - betas.index(beta)) / 0.03))
            iters_per_stage = min(iters_per_stage, 200)
            
            res = minimize(obj_func, best_x, method='L-BFGS-B',
                          options={'maxiter': iters_per_stage, 'ftol': 1e-12, 'gtol': 1e-8,
                                  'maxfun': iters_per_stage * 5})
            if res.x is not None and res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            pass
    
    cand = best_x.reshape(-1, 2)
    return cand


def _local_polish(pts, time_budget):
    """Final local polish using the exact min-triangle objective."""
    start_time = time.time()
    x0 = pts.flatten()
    
    best_x = x0.copy()
    best_score = _score(pts)
    
    # Use Nelder-Mead on the hard objective with tight tolerances
    try:
        res = minimize(_neg_score, best_x, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                              'adaptive': True, 'maxfev': 8000})
        if res.x is not None:
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_x = res.x
    except Exception:
        pass
    
    # Additional polish with multi-beta soft objective
    if time.time() - start_time < time_budget - 3:
        try:
            res = minimize(_multi_beta_soft_obj, best_x, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_x = res.x
        except Exception:
            pass
    
    # Final single-beta high-precision pass
    if time.time() - start_time < time_budget - 2:
        try:
            res = minimize(lambda x: _soft_obj(x, beta=500.0), best_x, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-11, 'fatol': 1e-13,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_x = res.x
        except Exception:
            pass
    
    return best_x.reshape(-1, 2)


def construct():
    start_time = time.time()
    
    # Generate diverse initial configurations
    inits = [
        _initial_regular_polygon(N),
        _initial_mixed(N),
        _initial_hexagonal(N),
        _initial_cosine_mod(N),
        _initial_grid_based(N),
    ]
    
    # Add perturbed versions of the regular polygon
    rng = np.random.RandomState(42)
    for seed in [1, 2, 3]:
        base = _initial_regular_polygon(N)
        noise = rng.normal(0, 0.08, size=base.shape)
        perturbed = base + noise
        perturbed = _normalize_unit_area(perturbed)
        inits.append(perturbed)
    
    # Evaluate initial scores
    init_scores = []
    for init in inits:
        s = _score(init)
        init_scores.append(s)
    
    # Sort by initial score (best first)
    sorted_indices = np.argsort(init_scores)[::-1]
    inits = [inits[i] for i in sorted_indices]
    
    best_pts = inits[0]
    best_score = init_scores[sorted_indices[0]]
    
    # Allocate time: ~55% for optimization, ~35% for polishing, ~10% buffer
    total_time = DEADLINE - (time.time() - start_time)
    opt_time = total_time * 0.55
    polish_time = total_time * 0.35
    
    # Optimize from top initializations
    for idx, init in enumerate(inits[:5]):
        if time.time() - start_time > DEADLINE - polish_time - 5:
            break
        
        time_left = DEADLINE - polish_time - 5 - (time.time() - start_time)
        if time_left < 2:
            break
        
        try:
            cand = _optimize_with_annealing(init, time_left * 0.9)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand
        except Exception:
            pass
    
    # Final polish
    if time.time() - start_time < DEADLINE - 3:
        try:
            polish_budget = min(polish_time, DEADLINE - 5 - (time.time() - start_time))
            if polish_budget > 1:
                best_pts = _local_polish(best_pts, polish_budget)
                final_score = _score(best_pts)
                if final_score > best_score:
                    best_score = final_score
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
