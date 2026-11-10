# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Initial guess: 4x4 grid
    x = np.arange(4)
    y = np.arange(4)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))

    def score(points):
        """Compute R = (d_min / d_max)^2"""
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(params):
        """
        Objective function to minimize: -d_min / d_max
        We maximize d_min/d_max, so we minimize its negative.
        """
        pts = params.reshape((n, 2))
        dists = pdist(pts, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-10:
            return 0.0
            
        ratio = d_min / d_max
        return -ratio

    best_points = points.copy()
    best_score = score(best_points)

    # Try local optimization from the grid start
    try:
        res = minimize(
            objective, 
            points.flatten(), 
            method='L-BFGS-B', 
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        
        if time.time() > deadline:
            return best_points
            
        candidate_points = res.x.reshape((n, 2))
        candidate_score = score(candidate_points)
        
        if candidate_score > best_score:
            best_points = candidate_points
            best_score = candidate_score
            
    except Exception:
        pass

    # If time allows, try a few random restarts or perturbations to escape local minima
    if time.time() < deadline - 60:
        # Try a hexagonal-like arrangement
        # 4 rows of 4 points, but staggered
        hex_points = np.array([
            [0, 0], [1, 0], [2, 0], [3, 0],
            [0.5, 0.866], [1.5, 0.866], [2.5, 0.866], [3.5, 0.866],
            [0, 1.732], [1, 1.732], [2, 1.732], [3, 1.732],
            [0.5, 2.598], [1.5, 2.598], [2.5, 2.598], [3.5, 2.598]
        ])
        
        try:
            res_hex = minimize(
                objective, 
                hex_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if time.time() > deadline:
                return best_points
                
            candidate_points = res_hex.x.reshape((n, 2))
            candidate_score = score(candidate_points)
            
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
                
        except Exception:
            pass

    # Random restarts with Powell
    np.random.seed(42)
    for i in range(10):
        if time.time() > deadline - 30:
            break
            
        # Random points in a square
        rand_pts = np.random.rand(n, 2) * 10 - 5
        try:
            res_rand = minimize(
                objective, 
                rand_pts.flatten(), 
                method='Powell', 
                options={'maxiter': 1000, 'ftol': 1e-15, 'xtol': 1e-15}
            )
            if time.time() > deadline:
                break
                
            cand = res_rand.x.reshape((n, 2))
            score_val = score(cand)
            if score_val > best_score:
                best_points = cand
                best_score = score_val
        except Exception:
            continue

    return best_points

# EVOLVE-BLOCK-END
