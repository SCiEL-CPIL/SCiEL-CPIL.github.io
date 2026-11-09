# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _score(points):
    """Calculate the score R = (d_min / d_max)^2."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max < 1e-10:
        return 0.0
    ratio = d_min / d_max
    return ratio ** 2

def _objective(points_flat):
    """Objective function for minimization (negative score)."""
    points = points_flat.reshape(16, 2)
    s = _score(points)
    return -s

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    best_score = -1.0
    best_points = None
    
    # Define initial configurations
    # 1. Regular polygon (often a good local optimum)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init1 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Random points in unit disk
    rng = np.random.default_rng(42)
    init2 = rng.uniform(-1, 1, size=(n, 2))
    norms = np.linalg.norm(init2, axis=1, keepdims=True)
    init2 = init2 / np.maximum(norms, 1e-8)
    
    inits = [init1, init2]
    
    for init in inits:
        x0 = init.flatten()
        
        # Powell method is derivative-free and handles non-smoothness of min-distance
        res = minimize(_objective, x0, method='Powell', 
                       options={'maxiter': 2000, 'maxfev': 50000, 'xtol': 1e-9, 'ftol': 1e-9})
        
        opt_points = res.x.reshape(n, 2)
        
        # Normalize: center and scale to unit max distance
        opt_points -= np.mean(opt_points, axis=0)
        d_max = np.max(pdist(opt_points))
        if d_max > 1e-10:
            opt_points /= d_max
            
        s = _score(opt_points)
        
        if s > best_score:
            best_score = s
            best_points = opt_points

    return best_points

# EVOLVE-BLOCK-END
