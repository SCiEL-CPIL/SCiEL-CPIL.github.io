import numpy as np
from scipy.optimize import differential_evolution
import time

def construct():
    n = 8
    # Regular octagon as fallback
    def regular_octagon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def score(points):
        # points: (n, 2)
        # Scale to diameter 2
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D == 0:
            return 0.0
        scaled_dists = dists * (2.0 / D)
        # Product of distances for i < j
        # Mask upper triangle
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        prod = np.prod(scaled_dists[mask])
        return prod / (n ** n)

    def objective(x):
        # x: (16,)
        pts = x.reshape(n, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Log product of distances (i < j)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        # Add small epsilon to avoid log(0)
        log_sum = np.sum(np.log(dists[mask] + 1e-12))
        
        # Diameter penalty
        D = np.max(dists)
        penalty = 0.0
        if D > 2.0:
            # Sum of squared excesses
            excess = np.maximum(dists - 2.0, 0.0)
            penalty = 10.0 * np.sum(excess**2)
            
        # Regularization to keep points near origin
        reg = 1e-4 * np.sum(pts**2)
        
        return -log_sum + penalty + reg

    start_time = time.time()
    timeout = 100  # seconds
    
    bounds = [(-2.0, 2.0)] * (2 * n)
    
    best_score = score(regular_octagon())
    best_points = regular_octagon()
    
    try:
        # Run differential evolution
        # maxiter and popsize can be adjusted based on time
        # We use a callback to stop if time is up
        def callback(xk, convergence):
            return time.time() - start_time > timeout
        
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=1000, 
            popsize=15, 
            tol=1e-8, 
            callback=callback,
            seed=42
        )
        
        if res.success or res.fun < objective(res.x): # res.x is the best found
            candidate_points = res.x.reshape(n, 2)
            # Center the points
            candidate_points -= np.mean(candidate_points, axis=0)
            
            # Check validity
            diffs = candidate_points[:, np.newaxis, :] - candidate_points[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            if np.all(np.diff(np.sort(dists[np.triu_indices(n, k=1)])) > 1e-6): # Distinct?
                # Actually just check if distinct enough
                min_dist = np.min(dists[mask])
                if min_dist > 1e-5:
                    cand_score = score(candidate_points)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_points = candidate_points

    except Exception as e:
        pass

    return best_points.tolist()
