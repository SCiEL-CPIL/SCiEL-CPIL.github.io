import numpy as np
from scipy.optimize import differential_evolution

def construct():
    n = 8
    
    # Objective function to minimize (negative log score)
    # Maximize sum(log(d_ij)) - n(n-1)*log(D)
    # Minimize -sum(log(d_ij)) + n(n-1)*log(D)
    def objective(params):
        pts = params.reshape(n, 2)
        # Calculate pairwise distances
        dists = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
        # Avoid log(0)
        dists[~np.isfinite(dists)] = 1e-10
        # Set diagonal to 1 to ignore self-distance
        np.fill_diagonal(dists, 1.0)
        
        # Diameter
        D = np.max(dists)
        if D < 1e-9:
            return 1e10 # Penalty for collapse
        
        # Sum of logs of all distinct pairs
        log_prod = np.sum(np.log(dists))
        
        # n(n-1) for n=8 is 56
        penalty = 56.0 * np.log(D)
        
        return -log_prod + penalty

    # Bounds for coordinates
    # The shape is scale-invariant, so we can bound the coordinates.
    # A diameter of 2 fits within a box of size 2x2.
    # We use slightly larger bounds to allow the optimizer to explore.
    bounds = [(-1.5, 1.5)] * (2 * n)
    
    # Run Differential Evolution
    # This is a global optimization method, distinct from the local Nelder-Mead approach.
    # It is robust for finding the global maximum of the product of distances.
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-6,
            polish=True
        )
        best_points = res.x.reshape(n, 2)
    except Exception:
        # Fallback to regular octagon if optimization fails
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return [[float(p[0]), float(p[1])] for p in best_points]
