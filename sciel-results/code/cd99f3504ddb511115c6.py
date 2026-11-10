import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    
    def objective(params):
        points = params.reshape(n, 2)
        # Center the points to remove translation invariance
        points = points - np.mean(points, axis=0)
        
        # Normalize scale to avoid scale invariance issues
        norm = np.linalg.norm(points)
        if norm < 1e-8:
            return 1e10
        points = points / norm
        
        # Calculate pairwise distances
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-8:
            return 1e10
            
        # We want to maximize (d_min / d_max)^2
        # So minimize -(d_min / d_max)^2
        ratio = d_min / d_max
        return -(ratio ** 2)

    # Generate initial guesses
    
    # Guess 1: 4x4 Grid
    x = np.arange(4) - 1.5
    y = np.arange(4) - 1.5
    xx, yy = np.meshgrid(x, y)
    guess1 = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Guess 2: Hexagonal lattice subset (16 closest to center)
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            hex_points.append(p)
    hex_points = np.array(hex_points)
    # Center
    hex_points = hex_points - np.mean(hex_points, axis=0)
    # Sort by distance from center, take 16 closest
    dists_to_center = np.linalg.norm(hex_points, axis=1)
    idx = np.argsort(dists_to_center)[:16]
    guess2 = hex_points[idx]
    
    # Guess 3: Random
    np.random.seed(42)
    guess3 = np.random.randn(n, 2)
    
    guesses = [guess1, guess2, guess3]
    
    best_score = -1e10
    best_points = None
    
    for g in guesses:
        # Flatten for optimizer
        x0 = g.flatten()
        
        # Run Nelder-Mead
        # maxfev to keep it fast
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxfev': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
            
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(n, 2)
        except Exception:
            pass
            
    # If optimization failed or didn't improve, fallback to best guess
    if best_points is None:
        best_points = guess1
        
    return best_points
