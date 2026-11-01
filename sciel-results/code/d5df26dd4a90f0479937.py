import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Initial configuration: Regular N-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
    
    best_score = -np.inf
    best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    def get_points(x):
        return x.reshape(-1, 2)

    def compute_scores(pts):
        # Compute convex hull area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume  # In 2D, volume is area
        except Exception:
            return -np.inf, np.zeros((N, 2))
        
        if hull_area <= 1e-10:
            return -np.inf, pts

        # Compute all triangle areas
        # n choose 3
        indices = list(itertools.combinations(range(N), 3))
        areas = np.zeros(len(indices))
        for idx, (i, j, k) in enumerate(indices):
            p1, p2, p3 = pts[i], pts[j], pts[k]
            # Area = 0.5 * |cross(p2-p1, p3-p1)|
            v1 = p2 - p1
            v2 = p3 - p1
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            areas[idx] = 0.5 * abs(cross)
        
        min_area = np.min(areas)
        score = min_area / hull_area
        return score, areas

    def objective(x):
        pts = get_points(x)
        # Check time
        if time.time() > deadline:
            return 0.0  # Return a neutral value, optimizer should stop soon
        
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            return 1e10 # Penalize invalid configurations
            
        if hull_area <= 1e-10:
            return 1e10

        indices = list(itertools.combinations(range(N), 3))
        areas = np.zeros(len(indices))
        for idx, (i, j, k) in enumerate(indices):
            p1, p2, p3 = pts[i], pts[j], pts[k]
            v1 = p2 - p1
            v2 = p3 - p1
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            areas[idx] = 0.5 * abs(cross)
        
        ratios = areas / hull_area
        
        # Soft-min to maximize
        # We want to maximize min(ratios).
        # Minimize -soft_min(ratios)
        beta = 100.0
        # Use log-sum-exp for stability
        max_r = np.max(ratios)
        log_sum = max_r + np.log(np.sum(np.exp(-beta * (ratios - max_r))))
        soft_min = max_r - np.log(np.sum(np.exp(-beta * (ratios - max_r)))) / beta
        
        return -soft_min

    # Multi-start with different initializations
    inits = [
        x0,
        np.column_stack((1.1 * np.cos(angles), 1.1 * np.sin(angles))).flatten(),
        np.column_stack((0.9 * np.cos(angles), 0.9 * np.sin(angles))).flatten(),
        np.column_stack((np.cos(angles + 0.1), np.sin(angles + 0.1))).flatten()
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(objective, init, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-8})
            
            if res.success or res.fun < 0:
                final_pts = get_points(res.x)
                score, _ = compute_scores(final_pts)
                if score > best_score:
                    best_score = score
                    best_points = final_pts
        except Exception:
            pass

    # Final check and return
    if time.time() > deadline:
         return best_points
         
    # If best_score is still -inf, fallback to initial
    if best_score == -np.inf:
        return best_points

    return best_points
