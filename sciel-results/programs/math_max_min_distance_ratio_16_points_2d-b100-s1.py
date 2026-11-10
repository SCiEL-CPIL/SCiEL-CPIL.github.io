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
    Smooth surrogate for optimization.
    We want to maximize min(dist) / max(dist).
    Using a power mean with high power approximates the max, and low power approximates min.
    To make it smooth and differentiable for DE/gradient methods, we use:
    S = (sum(d^p) / N)^(1/p) for min (p -> -inf)
    M = (sum(d^q) / N)^(1/q) for max (q -> +inf)
    We want to maximize S/M.
    Let's use p = -10, q = 10.
    Score = (S/M)^2 to match the metric scale roughly.
    """
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    
    # Clip distances to avoid division by zero or numerical issues
    d = np.clip(d, 1e-6, 1e6)
    
    p = 10.0
    # Power mean for min (p negative)
    # M_p = (sum(d^p)/N)^(1/p)
    # For min, we want p -> -inf. Let's use -10.
    min_mean = (np.sum(d ** (-p)) / len(d)) ** (-1.0 / p)
    # For max, we want p -> +inf. Let's use 10.
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
    Multi-stage local refinement: Powell -> Nelder-Mead -> Powell.
    """
    start = time.time()
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    # Stage 1: Powell
    def _neg_score(flat):
        pts = flat.reshape(-1, 2)
        return -_compute_score(pts)
        
    flat = current_points.ravel()
    
    try:
        res = minimize(_neg_score, flat, method='Powell',
                       options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-12})
        if res.fun < -current_score:
            current_points = res.x.reshape(-1, 2)
            current_score = -res.fun
    except Exception:
        pass
    
    if time.time() - start > time_budget:
        return current_points, current_score

    # Stage 2: Nelder-Mead
    try:
        res = minimize(_neg_score, current_points.ravel(), method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True})
        if res.fun < -current_score:
            current_points = res.x.reshape(-1, 2)
            current_score = -res.fun
    except Exception:
        pass
    
    if time.time() - start > time_budget:
        return current_points, current_score

    # Stage 3: Powell again for polish
    try:
        res = minimize(_neg_score, current_points.ravel(), method='Powell',
                       options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-15})
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
    deadline = start_time + 600  # safety margin before 630s kill
    
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
    # DE is good at global search but can be slow. We use a smooth objective to guide it.
    # Bounds: points should be within a reasonable range.
    # Normalizing scale doesn't affect the ratio, but helps DE.
    # Let's bound coordinates to [-2, 2].
    bounds = [(-2.0, 2.0)] * 32
    
    # Time budget for DE: 300 seconds
    de_time_budget = 300
    de_start = time.time()
    
    try:
        # Use the smooth objective for DE
        # DE minimizes, so we pass negative of the smooth score
        # Note: DE in scipy uses a population, which is expensive. 
        # We limit maxiter and use tol to stop early.
        res = differential_evolution(
            _neg_smooth_objective, 
            bounds, 
            seed=42,
            maxiter=200, 
            tol=1e-6,
            popsize=15,
            polish=False # We will do our own refinement
        )
        
        if res.fun < 0: # Should be negative since we negated
            candidate_points = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate_points)
            _update(candidate_points, candidate_score)
    except Exception as e:
        pass # If DE fails, fall back to local search from initial
    
    # Check time
    if time.time() > deadline - 100:
         # If we are running out of time, just return best so far or initial
         if best_points is None:
             best_points = initial_points
             best_score = _compute_score(best_points)
         return best_points

    # Phase 2: Local Refinement of the best candidate found by DE
    if best_points is not None:
        refined_points, refined_score = _local_refine(best_points, time_budget=100, rng=rng)
        _update(refined_points, refined_score)
        
    # Phase 3: Additional random restarts with local refinement if time permits
    # This helps if DE got stuck in a local minimum of the smooth surrogate
    # that is not the global optimum of the true metric.
    while time.time() < deadline - 20:
        # Generate a random point set
        # We can perturb the best known solution or generate fresh
        # Perturbing is often more effective
        if best_points is not None:
            noise_scale = 0.1
            new_points = best_points + rng.normal(0, noise_scale, size=(16, 2))
        else:
            new_points = rng.uniform(-1.5, 1.5, size=(16, 2))
            
        # Quick local refinement
        # Use a shorter time budget for these restarts
        time_left = deadline - time.time() - 10
        if time_left < 5:
            break
            
        # Allocate small chunk of time
        alloc_time = min(10, time_left / 2)
        
        refined_points, refined_score = _local_refine(new_points, time_budget=alloc_time, rng=rng)
        _update(refined_points, refined_score)
        
        # If we found a significantly better score, we might want to restart the noise scale
        # but for simplicity, we keep it constant.
        
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
