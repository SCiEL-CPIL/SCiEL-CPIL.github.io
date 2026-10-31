import numpy as np
from scipy.optimize import minimize

def construct():
    N = 16
    
    # Objective function: Minimize (max_d / min_d)^2
    def objective(x):
        pts = x.reshape(N, 2)
        # Compute pairwise distances
        dist_matrix = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        iu = np.triu_indices(N, 1)
        pairwise_dists = dist_matrix[iu]
        
        min_d = np.min(pairwise_dists)
        max_d = np.max(pairwise_dists)
        
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d) ** 2

    best_score = np.inf
    best_x = None
    
    # Define several initializers to provide diverse starting points
    
    # 1. Concentric Rings (1 center, 5 inner, 10 outer)
    def init_rings():
        pts = []
        pts.append([0, 0])
        for i in range(5):
            theta = 2 * np.pi * i / 5
            pts.append([np.cos(theta), np.sin(theta)])
        for i in range(10):
            theta = 2 * np.pi * i / 10 + np.pi/10
            pts.append([2 * np.cos(theta), 2 * np.sin(theta)])
        return np.array(pts).flatten()

    # 2. Fibonacci Spiral
    def init_spiral():
        pts = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(N):
            r = np.sqrt(i + 1)
            theta = i * golden_angle
            pts.append([r * np.cos(theta), r * np.sin(theta)])
        return np.array(pts).flatten()

    # 3. Square Grid
    def init_grid():
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i - 1.5, j - 1.5])
        return np.array(pts).flatten()

    # 4. Random
    def init_random(seed=42):
        rng = np.random.RandomState(seed)
        pts = rng.uniform(-5, 5, (N, 2))
        return pts.flatten()

    initializers = [
        init_rings, 
        init_spiral, 
        init_grid, 
        init_random, 
        lambda: init_random(123), 
        lambda: init_random(999)
    ]

    for init_func in initializers:
        try:
            x0 = init_func()
            # Center the points
            pts0 = x0.reshape(N, 2)
            pts0 -= np.mean(pts0, axis=0)
            x0 = pts0.flatten()
            
            # Use Powell method for robust local convergence
            result = minimize(
                objective, 
                x0, 
                method='Powell',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-8
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x
        except Exception:
            continue
            
    if best_x is None:
        # Fallback
        return np.random.uniform(-5, 5, (N, 2))
        
    # Re-center
    pts = best_x.reshape(N, 2)
    pts -= np.mean(pts, axis=0)
    
    return pts
