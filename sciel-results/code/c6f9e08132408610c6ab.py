import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    
    def generate_points(params):
        # params: [x1, y1, x2, y2, x3, y3, x4, y4]
        # Generate 16 points using D2 symmetry (reflections across x and y axes)
        # Orbits:
        # (x1, y1) -> (x1, y1), (-x1, y1), (x1, -y1), (-x1, -y1)
        # (x2, y2) -> (x2, y2), (-x2, y2), (x2, -y2), (-x2, -y2)
        # (x3, y3) -> (x3, y3), (-x3, y3), (x3, -y3), (-x3, -y3)
        # (x4, y4) -> (x4, y4), (-x4, y4), (x4, -y4), (-x4, -y4)
        
        pts = np.zeros((16, 2))
        
        for i in range(4):
            x = params[2*i]
            y = params[2*i + 1]
            
            # Ensure positive to avoid duplicates on axes if possible, 
            # though DE bounds will handle it. 
            # If x or y is 0, we get fewer unique points, leading to d_min=0.
            # We want to avoid axes.
            
            idx = 4 * i
            pts[idx, 0] = x
            pts[idx, 1] = y
            
            pts[idx+1, 0] = -x
            pts[idx+1, 1] = y
            
            pts[idx+2, 0] = x
            pts[idx+2, 1] = -y
            
            pts[idx+3, 0] = -x
            pts[idx+3, 1] = -y
            
        return pts

    def objective(flat_params):
        pts = generate_points(flat_params)
        # Check for duplicates (should be rare if bounds are > 0)
        dists = pdist(pts)
        if dists.size == 0:
            return 1e10
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return 1e10
        if d_min < 1e-12:
            return 1e10
            
        ratio = d_min / d_max
        return - (ratio ** 2)

    # Initial guess based on 4x4 grid in first quadrant
    # Grid points: (0.5, 0.5), (0.5, 1.5), (1.5, 0.5), (1.5, 1.5)
    x0 = [0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 1.5, 1.5]
    
    # Bounds for x and y. Since scale is invariant, we can pick any range.
    # Let's use [0.1, 3.0] to allow for significant distortion.
    bounds = [(0.1, 3.0)] * 8
    
    best_result = None
    best_score = -np.inf
    
    # Run multiple DE runs with different seeds to escape local minima
    for seed in [42, 123, 456, 789, 101]:
        try:
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=300, 
                popsize=20, 
                tol=1e-8, 
                seed=seed,
                polish=True,
                init='latinhypercube'
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
                
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback to the grid
        return np.array([
            [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5],
            [0.5, 1.5], [-0.5, 1.5], [0.5, -1.5], [-0.5, -1.5],
            [1.5, 0.5], [-1.5, 0.5], [1.5, -0.5], [-1.5, -0.5],
            [1.5, 1.5], [-1.5, 1.5], [1.5, -1.5], [-1.5, -1.5]
        ])

    best_points = generate_points(best_result)
    
    # Final local polish with Powell on the full 32D space? 
    # Or stick to 8D? 
    # Powell on 8D is fast.
    result_powell = minimize(
        objective, 
        best_result, 
        method='Powell', 
        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
    )
    
    if result_powell.fun < best_score:
        best_points = generate_points(result_powell.x)

    # Verify shape and finiteness
    if not np.all(np.isfinite(best_points)):
        return np.array([
            [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5],
            [0.5, 1.5], [-0.5, 1.5], [0.5, -1.5], [-0.5, -1.5],
            [1.5, 0.5], [-1.5, 0.5], [1.5, -0.5], [-1.5, -0.5],
            [1.5, 1.5], [-1.5, 1.5], [1.5, -1.5], [-1.5, -1.5]
        ])

    return best_points
