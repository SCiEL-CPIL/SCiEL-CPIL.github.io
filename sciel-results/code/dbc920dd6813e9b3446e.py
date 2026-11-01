import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

N = 14
D = 3

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    def objective(params):
        # params is shape (N, D)
        # Center the points
        pts = params - np.mean(params, axis=0)
        # Calculate pairwise squared distances
        # Using broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Exclude diagonal
        np.fill_diagonal(dist_sq, np.inf)
        d_max = np.max(dist_sq)
        d_min = np.min(dist_sq)
        
        if d_min < 1e-12:
            return 1e12
        return d_max / d_min

    def normalize(pts):
        # Normalize to unit RMS radius
        center = np.mean(pts, axis=0)
        pts = pts - center
        norm = np.sqrt(np.sum(pts**2))
        if norm > 1e-12:
            pts = pts / norm
        return pts

    best_score = np.inf
    best_pts = None
    
    # Strategy 1: Differential Evolution
    # Bounds: [-1, 1] for each coordinate
    bounds = [(-1, 1)] * (N * D)
    
    # We can try DE first, it's robust
    try:
        # Limit iterations to ensure it finishes
        # popsize=10, maxiter=50 might be enough for a rough solution
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10,
            tol=1e-4,
            polish=False # Polish with Nelder-Mead later if time permits
        )
        if time.time() < deadline:
            current_pts = res.x.reshape(N, D)
            current_pts = normalize(current_pts)
            score = objective(current_pts)
            if score < best_score:
                best_score = score
                best_pts = current_pts
    except Exception:
        pass

    # Strategy 2: Multi-start Nelder-Mead / Powell
    # If DE didn't find a good solution or if we have time
    # Let's do a few local optimizations from random starts
    
    num_starts = 20
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        # Random initialization on a sphere
        init_pts = np.random.randn(N, D)
        init_pts = normalize(init_pts)
        
        # Use Powell's method which is robust for non-smooth
        try:
            res = minimize(
                lambda x: objective(x.reshape(N, D)),
                init_pts.flatten(),
                method='Powell',
                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if time.time() > deadline:
                break
            current_pts = res.x.reshape(N, D)
            current_pts = normalize(current_pts)
            score = objective(current_pts)
            if score < best_score:
                best_score = score
                best_pts = current_pts
        except Exception:
            continue

    if best_pts is None:
        # Fallback to grid if optimization failed
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
        best_pts = normalize(best_pts)

    return best_pts
