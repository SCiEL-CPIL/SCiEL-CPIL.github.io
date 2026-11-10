# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses multi-start optimization of the sum of log distances (geometric mean) with scale normalization.
    """
    n = 16
    d = 2
    deadline = time.time() + 590  # Safety margin
    
    def normalize(points):
        # Fix scale: sum of squared norms = 1
        # Fix centroid: mean = 0
        points = points - np.mean(points, axis=0)
        norm_sq = np.sum(points**2)
        if norm_sq > 1e-12:
            points = points / np.sqrt(norm_sq)
        else:
            # Fallback if degenerate
            points = np.random.randn(n, d)
            points = points - np.mean(points, axis=0)
            points = points / np.sqrt(np.sum(points**2))
        return points

    def objective_flat(flat_points):
        if time.time() > deadline:
            return 0
        points = flat_points.reshape(n, d)
        # Calculate pairwise distances
        dists = pdist(points)
        # Avoid log(0)
        dists = np.maximum(dists, 1e-12)
        # Maximize sum of logs -> Minimize negative sum of logs
        return -np.sum(np.log(dists))

    def constraint_func(flat_points):
        points = flat_points.reshape(n, d)
        # We want sum of squares = 1
        return np.sum(points**2) - 1.0

    constraints = [{'type': 'eq', 'fun': constraint_func}]
    
    best_score = -np.inf
    best_points = None
    
    # Generate initial guesses
    inits = []
    
    # 1. Regular 16-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(p1)
    
    # 2. 4x4 Grid
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    p2 = np.column_stack((xx.ravel(), yy.ravel()))
    inits.append(p2)
    
    # 3. Random
    np.random.seed(42)
    p3 = np.random.randn(n, d)
    inits.append(p3)
    
    # 4. Another random
    np.random.seed(123)
    p4 = np.random.randn(n, d)
    inits.append(p4)
    
    # 5. Two concentric squares (8+8)
    ang1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    p5_1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
    ang2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    p5_2 = np.column_stack([np.cos(ang2), np.sin(ang2)]) * 0.5
    p5 = np.vstack([p5_1, p5_2])
    inits.append(p5)

    for init in inits:
        if time.time() > deadline:
            break
            
        # Normalize initial guess
        init_norm = normalize(init)
        flat_init = init_norm.ravel()
        
        # Run optimization
        try:
            res = minimize(
                objective_flat,
                flat_init,
                method='L-BFGS-B',
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res.success:
                final_points = res.x.reshape(n, d)
                # Final normalization to ensure strict compliance
                final_points = normalize(final_points)
                
                # Calculate true score
                dists = pdist(final_points)
                d_min = np.min(dists)
                d_max = np.max(dists)
                
                if d_max > 1e-12:
                    score = (d_min / d_max) ** 2
                else:
                    score = 0
                    
                if score > best_score:
                    best_score = score
                    best_points = final_points
                    
        except Exception:
            pass

    # Fallback if optimization failed or didn't run
    if best_points is None:
        best_points = normalize(inits[0])
        
    return best_points

# EVOLVE-BLOCK-END
