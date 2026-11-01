import numpy as np
from scipy.optimize import minimize
import time

N = 14
D = 3

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    def score(points):
        # points: (N, D)
        # Calculate pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dists = dists[mask]
        if len(dists) == 0:
            return 1e10
        min_d = np.min(dists)
        max_d = np.max(dists)
        if min_d < 1e-6:
            return 1e10
        return (max_d / min_d) ** 2

    def objective(x):
        points = x.reshape(N, D)
        return score(points)

    best_score = float('inf')
    best_points = None

    # Initializations
    inits = []
    
    # 1. Random points in a unit cube
    rng = np.random.default_rng(42)
    for _ in range(5):
        inits.append(rng.random((N, D)))
        
    # 2. Points on a sphere
    for _ in range(5):
        pts = rng.normal(size=(N, D))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
        
    # 3. Grid-like
    # 14 points in a 3x3x3 grid? 27 points. Pick 14.
    grid_pts = np.array(list(__import__('itertools').product(range(3), repeat=3)))[:14].astype(float)
    grid_pts = (grid_pts - grid_pts.mean(axis=0)) / np.linalg.norm(grid_pts, axis=1).max()
    inits.append(grid_pts)
    
    # 4. Tammes-like (approximate)
    # Just use random sphere points with some perturbation

    for init in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective, 
                init.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(N, D)
        except Exception as e:
            pass

    if best_points is None:
        # Fallback to seed
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_points = np.array(grid, dtype=float)
        best_score = score(best_points)

    return best_points
