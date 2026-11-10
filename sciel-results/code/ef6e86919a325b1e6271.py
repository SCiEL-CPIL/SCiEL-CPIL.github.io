# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses differential evolution to globally optimize the point positions.
    """
    n = 16
    d = 2
    
    def objective(params):
        pts = params.reshape(n, d)
        
        # Center the points to remove translation invariance
        pts = pts - np.mean(pts, axis=0)
        
        # Calculate pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu]
        
        d_max = np.max(pair_dists)
        
        if d_max < 1e-9:
            return 0.0
            
        # Normalize so that d_max = 1. 
        # This makes the problem scale-invariant and allows us to just maximize d_min.
        pair_dists_norm = pair_dists / d_max
        d_min_norm = np.min(pair_dists_norm)
        
        # We want to maximize d_min_norm, so we minimize -d_min_norm
        # The score R is (d_min / d_max)^2 = d_min_norm^2
        # Minimizing -d_min_norm is monotonic with maximizing R.
        
        return -d_min_norm

    # Define bounds for the points. 
    # Since we normalize, the scale doesn't matter much, but we need bounds for the optimizer.
    # A reasonable bound is [-5, 5] to allow for various shapes before normalization.
    bounds = [(-5.0, 5.0)] * (n * d)
    
    # Run differential evolution
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=1000, 
            popsize=30, 
            tol=1e-6,
            atol=1e-6,
            seed=42,
            init='latinhypercube'
        )
        
        best_points = res.x.reshape(n, d)
        
    except Exception as e:
        # Fallback to a simple grid if optimization fails
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack((xx.ravel(), yy.ravel()))
        
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
         x = np.linspace(-1.5, 1.5, 4)
         y = np.linspace(-1.5, 1.5, 4)
         xx, yy = np.meshgrid(x, y)
         best_points = np.column_stack((xx.ravel(), yy.ravel()))
         
    return best_points

# EVOLVE-BLOCK-END
