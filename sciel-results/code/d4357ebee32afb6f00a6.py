# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _smooth_objective(points):
    """
    Smooth surrogate for optimization using power means.
    """
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    
    d = np.clip(d, 1e-6, 1e6)
    
    p = 10.0
    min_mean = (np.sum(d ** (-p)) / len(d)) ** (-1.0 / p)
    max_mean = (np.sum(d ** p) / len(d)) ** (1.0 / p)
    
    if max_mean == 0:
        return 0.0
    
    ratio = min_mean / max_mean
    return ratio ** 2

def _neg_smooth_objective(flat):
    points = flat.reshape(-1, 2)
    return -_smooth_objective(points)

def _local_refine(points, time_budget, rng):
    """
    Multi-stage local refinement using smooth objective for gradient-based methods.
    """
    start = time.time()
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    def _neg_true_score(flat):
        pts = flat.reshape(-1, 2)
        return -_compute_score(pts)
    
    flat = current_points.ravel()
    
    # Stage 1: Powell with smooth objective
    try:
        res = minimize(_neg_smooth_objective, flat, method='Powell',
                       options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-12})
        if res.fun < -_smooth_objective(current_points):
            new_pts = res.x.reshape(-1, 2)
            new_score = _compute_score(new_pts)
            if new_score > current_score:
                current_points = new_pts
                current_score = new_score
    except Exception:
        pass
    
    if time.time() - start > time_budget * 0.4:
        return current_points, current_score

    # Stage 2: Nelder-Mead with smooth objective
    try:
        res = minimize(_neg_smooth_objective, current_points.ravel(), method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True})
        if res.fun < -_smooth_objective(current_points):
            new_pts = res.x.reshape(-1, 2)
            new_score = _compute_score(new_pts)
            if new_score > current_score:
                current_points = new_pts
                current_score = new_score
    except Exception:
        pass
    
    if time.time() - start > time_budget * 0.7:
        return current_points, current_score

    # Stage 3: Powell with true score for final polish
    try:
        res = minimize(_neg_true_score, current_points.ravel(), method='Powell',
                       options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-15})
        if res.fun < -current_score:
            current_points = res.x.reshape(-1, 2)
            current_score = -res.fun
    except Exception:
        pass
        
    return current_points, current_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = 0.0
    
    def _update(pts, sc):
        nonlocal best_points, best_score
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
            
    # Initial guess: 4x4 grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    initial_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Phase 1: Differential Evolution with smooth surrogate
    bounds = [(-2.0, 2.0)] * 32
    
    de_time_budget = 250
    de_start = time.time()
    
    try:
        res = differential_evolution(
            _neg_smooth_objective, 
            bounds, 
            seed=42,
            maxiter=300, 
            tol=1e-7,
            popsize=20,
            polish=False
        )
        
        if res.fun < 0:
            candidate_points = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate_points)
            _update(candidate_points, candidate_score)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 80:
        if best_points is None:
            best_points = initial_points
            best_score = _compute_score(best_points)
        return best_points

    # Phase 2: Local Refinement of the best candidate found by DE
    if best_points is not None:
        refined_points, refined_score = _local_refine(best_points, time_budget=80, rng=rng)
        _update(refined_points, refined_score)
        
    # Phase 3: Additional random restarts with local refinement
    restart_configs = [
        'grid', 'circle', 'random', 'perturbed'
    ]
    
    for config in restart_configs:
        if time.time() > deadline - 15:
            break
        
        if config == 'grid':
            x = np.linspace(-1.5, 1.5, 4)
            y = np.linspace(-1.5, 1.5, 4)
            xx, yy = np.meshgrid(x, y)
            new_points = np.column_stack([xx.ravel(), yy.ravel()])
        elif config == 'circle':
            angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
            radii = np.array([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5,
                              2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5])
            new_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        elif config == 'random':
            new_points = rng.uniform(-1.5, 1.5, size=(16, 2))
        elif config == 'perturbed':
            if best_points is not None:
                new_points = best_points + rng.normal(0, 0.15, size=(16, 2))
            else:
                new_points = rng.uniform(-1.5, 1.5, size=(16, 2))
        
        time_left = deadline - time.time() - 10
        if time_left < 5:
            break
            
        alloc_time = min(15, time_left / 2)
        
        refined_points, refined_score = _local_refine(new_points, time_budget=alloc_time, rng=rng)
        _update(refined_points, refined_score)
        
    # Final check
    if best_points is None:
        best_points = initial_points
        best_score = _compute_score(best_points)
        
    # Ensure finite and correct shape
    if not np.all(np.isfinite(best_points)):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack([xx.ravel(), yy.ravel()])
        
    return best_points

# EVOLVE-BLOCK-END
