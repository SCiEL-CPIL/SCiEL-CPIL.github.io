import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    
    def objective(flat_points):
        points = flat_points.reshape(n, 3)
        # Project back to unit sphere to handle numerical drift
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-10
        points = points / norms
        
        # Calculate pairwise distances
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        
        # If any distance is very small, return a large penalty
        if np.min(dists) < 1e-6:
            return 1e10
            
        # Sum of inverse squared distances
        # This encourages points to be as far apart as possible
        return np.sum(1.0 / (dists ** 2))

    best_points = None
    best_score = -1.0
    best_obj = np.inf
    
    n_restarts = 20
    
    for r in range(n_restarts):
        # Random initialization on sphere
        if r == 0:
            # Start with Fibonacci sphere for a good initial guess
            phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
            theta = np.pi * (1 + 5**0.5) * np.arange(n)
            points = np.column_stack((
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ))
        else:
            # Random points
            v = np.random.normal(size=(n, 3))
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            points = v / norms
            
        flat_points = points.flatten()
        
        # Use Powell method, which is derivative-free and robust
        # maxiter is set high enough to allow convergence
        res = minimize(objective, flat_points, method='Powell', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'xtol': 1e-12})
        
        opt_points = res.x.reshape(n, 3)
        # Final projection
        norms = np.linalg.norm(opt_points, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        opt_points = opt_points / norms
        
        # Calculate the actual metric for this configuration
        dists = pdist(opt_points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            score = 0.0
        else:
            score = (d_min / d_max) ** 2
            
        if score > best_score:
            best_score = score
            best_points = opt_points.copy()

    return best_points
