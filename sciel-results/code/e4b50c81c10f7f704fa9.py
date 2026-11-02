import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for triangle areas
_i, _j, _k = np.indices((N, N, N))
_mask = (_i < _j) & (_j < _k)
_IDX_I = _i[_mask]
_IDX_J = _j[_mask]
_IDX_K = _k[_mask]

def _compute_min_triangle_area(points):
    p_i = points[_IDX_I]
    p_j = points[_IDX_J]
    p_k = points[_IDX_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    return np.min(areas)

def _compute_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _objective(points_flat):
    points = points_flat.reshape(N, 2)
    # Points are constrained to the unit disk; hull area <= 1
    # Score = min_area / hull_area >= min_area (since hull_area <= 1)
    # So maximizing min_area in the unit disk is a lower bound on the true score.
    # For points on the boundary, hull_area = 1 and score = min_area exactly.
    min_area = _compute_min_triangle_area(points)
    return -min_area

def _soft_min_objective(points_flat, beta):
    points = points_flat.reshape(N, 2)
    p_i = points[_IDX_I]
    p_j = points[_IDX_J]
    p_k = points[_IDX_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    max_area = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_area))
    soft_min = -max_area - (1.0/beta) * np.log(np.sum(exp_terms))
    
    return -soft_min

def _constrain_to_disk(points):
    """Project points into the unit disk."""
    dists = np.linalg.norm(points, axis=1)
    mask = dists > 1.0
    if np.any(mask):
        points = points.copy()
        points[mask] /= dists[mask][:, None]
    return points

def _de_objective(points_flat):
    """Differential evolution objective: maximize min triangle area in unit disk."""
    points = _constrain_to_disk(points_flat.reshape(N, 2))
    min_area = _compute_min_triangle_area(points)
    return -min_area

def _local_refine(x0, deadline, rng):
    """Local refinement from a starting point."""
    x_current = x0.copy()
    
    # Step 1: Powell
    try:
        res = minimize(_objective, x_current, method='Powell', options={'maxiter': 300, 'maxfev': 3000})
        if time.time() > deadline:
            return x_current
        x_current = res.x
    except Exception:
        pass
    
    # Step 2: Soft-min annealing with L-BFGS-B
    betas = [20, 100, 500, 2000]
    for beta in betas:
        if time.time() > deadline:
            break
        try:
            res = minimize(lambda x: _soft_min_objective(x, beta), x_current, 
                         method='L-BFGS-B', options={'maxiter': 80, 'maxfev': 800, 'ftol': 1e-14, 'gtol': 1e-14})
            if time.time() > deadline:
                break
            x_current = res.x
        except Exception:
            pass
    
    # Step 3: Final Powell polish
    if time.time() < deadline - 2.0:
        try:
            res = minimize(_objective, x_current, method='Powell', options={'maxiter': 200, 'maxfev': 2000})
            x_current = res.x
        except Exception:
            pass
    
    return x_current

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    def _try_update(x_flat):
        nonlocal best_score, best_points
        pts = _constrain_to_disk(x_flat.reshape(N, 2))
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
        return score
    
    # Phase 1: Differential Evolution for global exploration
    # Bounds: each coordinate in [-1, 1] (unit square containing unit disk)
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    de_start = time.time()
    de_budget = 45.0  # seconds for DE
    
    try:
        from scipy.optimize import differential_evolution
        
        de_result = differential_evolution(
            _de_objective,
            bounds,
            seed=42,
            maxiter=80,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=False,
            update_interval=5,
            callback=None
        )
        
        if de_result.success or de_result.fun < 0:
            _try_update(de_result.x)
    except Exception:
        pass
    
    # Phase 2: Local refinement from multiple diverse starting points
    inits = []
    
    # 1. Regular 14-gon (on unit circle)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Jittered regular 14-gon
    for jitter_std in [0.05, 0.1, 0.15, 0.25]:
        jitter = rng.normal(0, jitter_std, (N, 2))
        pts = np.column_stack([np.cos(angles) + jitter[:,0], np.sin(angles) + jitter[:,1]])
        inits.append(_constrain_to_disk(pts))
    
    # 3. Two concentric 7-gons (alternating radii)
    for r_inner in [0.4, 0.5, 0.6]:
        pts = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 1.0 if i % 2 == 0 else r_inner
            pts[i] = [r * np.cos(angle), r * np.sin(angle)]
        inits.append(pts)
    
    # 4. Random points on/near unit circle
    for _ in range(8):
        angles_r = np.sort(rng.uniform(0, 2*np.pi, N))
        radii = rng.uniform(0.7, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles_r), radii * np.sin(angles_r)])
        inits.append(pts)
    
    # 5. Random points in disk
    for _ in range(5):
        pts = rng.random((N, 2)) * 2 - 1
        dists = np.linalg.norm(pts, axis=1)
        max_dist = np.max(dists)
        if max_dist > 0:
            pts = pts / max_dist
        inits.append(_constrain_to_disk(pts))
    
    # 6. If DE found something good, use it and its perturbations
    if best_points is not None:
        inits.append(best_points.copy())
        for pert_std in [0.05, 0.1]:
            pert = best_points + rng.normal(0, pert_std, (N, 2))
            inits.append(_constrain_to_disk(pert))
    
    # Refine each starting point
    for init in inits:
        if time.time() > deadline:
            break
        
        x0 = init.flatten()
        x_result = _local_refine(x0, deadline, rng)
        _try_update(x_result)
    
    # Phase 3: If time remains, do more DE with tighter bounds around best
    if best_points is not None and time.time() < deadline - 10.0:
        try:
            from scipy.optimize import differential_evolution
            
            # Tighter bounds around best solution
            tight_bounds = []
            for i in range(2 * N):
                center = best_points.flatten()[i]
                tight_bounds.append((center - 0.5, center + 0.5))
            
            de_result2 = differential_evolution(
                _de_objective,
                tight_bounds,
                seed=123,
                maxiter=50,
                popsize=12,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                update_interval=5
            )
            
            if de_result2.success:
                _try_update(de_result2.x)
        except Exception:
            pass
    
    # Final: ensure best_points is valid
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
