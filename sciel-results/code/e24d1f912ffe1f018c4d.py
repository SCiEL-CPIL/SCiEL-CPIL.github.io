import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist

_DEADLINE_SECONDS = 630.0
_SAFETY_MARGIN = 30.0

def _raw_score(points):
    """
    Calculate the score: (d_min / d_max)^2.
    Returns 0 if invalid.
    """
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0

    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    dmin = np.min(dists)
    dmax = np.max(dists)
    if not (np.isfinite(dmin) and np.isfinite(dmax)):
        return 0.0
    if dmax <= 1e-15 or dmin <= 1e-15:
        return 0.0
    r = dmin / dmax
    return float(r * r)

def _objective_cartesian(x):
    """
    Objective function for Cartesian coordinates.
    Minimizes -log(d_min/d_max) = log(d_max) - log(d_min).
    """
    X = x.reshape(14, 3)
    dists = pdist(X)
    dmin = np.min(dists)
    dmax = np.max(dists)
    
    if dmin <= 1e-15 or dmax <= 1e-15:
        return 1e10
        
    return np.log(dmax) - np.log(dmin)

def _soft_objective_cartesian(x, temperature=0.1):
    """
    Smoothed objective for better gradient estimation if needed, 
    but DE doesn't need gradients. 
    For Nelder-Mead polish, the raw objective is often better if initialized well.
    """
    X = x.reshape(14, 3)
    dists = pdist(X)
    dmin = np.min(dists)
    dmax = np.max(dists)
    
    if dmin <= 1e-15 or dmax <= 1e-15:
        return 1e10
        
    # Soft min/max approximation
    # log(sum(exp(-d/T))) / n is approx log(min(d))
    # This is complex. Let's stick to raw for NM if DE provides a good start.
    return np.log(dmax) - np.log(dmin)

def min_max_dist_dim3_14():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS - _SAFETY_MARGIN

    best_pts = None
    best_score = 0.0

    # Differential Evolution on Cartesian coordinates
    # Bounds: [-1, 1] for each coordinate
    bounds = [(-1.0, 1.0)] * 42
    
    # DE parameters
    # popsize=20 is standard, maybe 15 to save time
    # maxiter: DE converges slowly. 100-200 iterations might be enough for a good start.
    # We will run DE, then polish.
    
    try:
        # Use a seed for reproducibility
        de_result = differential_evolution(
            _objective_cartesian,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-8,
            init='latinhypercube',
            polish=False, # We will do our own polish
            updating='immediate',
            workers=1 # Single process to avoid overhead in this context
        )
        
        if de_result.success or de_result.x is not None:
            x_de = de_result.x
            # Polish with Nelder-Mead
            # NM is derivative-free and good for non-smooth functions
            nm_result = minimize(
                _objective_cartesian,
                x_de,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if nm_result.x is not None:
                x_polished = nm_result.x
                pts = x_polished.reshape(14, 3)
                score = _raw_score(pts)
                
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
                    
    except Exception as e:
        # If DE fails, fall back to random search or previous best
        pass

    # If we still have time, try a few more random starts with NM polish
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        for i in range(5):
            if time.time() > deadline - 5:
                break
            # Random start in [-1, 1]^42
            x_rand = rng.uniform(-1, 1, 42)
            
            # Quick NM polish
            try:
                nm_result = minimize(
                    _objective_cartesian,
                    x_rand,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if nm_result.x is not None:
                    pts = nm_result.x.reshape(14, 3)
                    score = _raw_score(pts)
                    if score > best_score:
                        best_score = score
                        best_pts = pts.copy()
            except:
                continue

    if best_pts is None:
        # Fallback: Fibonacci sphere
        n = 14
        golden_ratio = (1 + np.sqrt(5)) / 2
        theta = np.pi * (1 - 2 * (np.arange(n) + 0.5) / n)
        phi = 2 * np.pi * np.arange(n) / golden_ratio
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        best_pts = np.column_stack((x, y, z))
        best_score = _raw_score(best_pts)

    return best_pts
