import numpy as np
from scipy.optimize import differential_evolution
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

    # Define bounds for differential evolution
    # The actual scale/position doesn't matter due to normalization,
    # so we can search in a reasonable box like [-10, 10]
    bounds = [(-10.0, 10.0)] * (n * 2)
    
    # Run Differential Evolution
    # popsize=30, maxiter=1000, tol=1e-6, atol=1e-8
    # seed=42 for reproducibility
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=500, 
            popsize=30, 
            tol=1e-6, 
            atol=1e-8,
            seed=42,
            init='sobol' # Use Sobol sequence for better coverage
        )
        
        if res.success:
            best_points = res.x.reshape(n, 2)
        else:
            # Even if not successful, res.x is often a good candidate
            best_points = res.x.reshape(n, 2)
            
    except Exception as e:
        # Fallback to a grid if DE fails
        x = np.arange(4) - 1.5
        y = np.arange(4) - 1.5
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack((xx.ravel(), yy.ravel()))
        
    return best_points
