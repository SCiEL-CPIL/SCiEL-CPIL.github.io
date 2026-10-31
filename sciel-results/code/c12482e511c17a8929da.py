import numpy as np
from scipy.optimize import basinhopping, minimize
import random

def construct():
    n_points = 16
    
    def objective(coords):
        pts = coords.reshape(-1, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min < 1e-6:
            return 1e10
        return (d_max / d_min) ** 2

    best_score = float('inf')
    best_coords = None
    
    # Start 1: Concentric rings (1, 6, 9)
    pts1 = [[0.0, 0.0]]
    for i in range(6):
        theta = i * np.pi / 3
        pts1.append([1.0 * np.cos(theta), 1.0 * np.sin(theta)])
    for i in range(9):
        theta = i * 2 * np.pi / 9
        pts1.append([2.0 * np.cos(theta), 2.0 * np.sin(theta)])
    init1 = np.array(pts1).flatten()
    
    # Start 2: Random
    random.seed(123)
    init2 = np.random.uniform(-2, 2, size=(n_points, 2)).flatten()
    
    # Start 3: Regular 16-gon (distorted)
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    init3 = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
    
    for init in [init1, init2, init3]:
        try:
            res = basinhopping(objective, init, 
                               niter=50, 
                               minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 1000}},
                               seed=42)
            if res.fun < best_score:
                best_score = res.fun
                best_coords = res.x
        except Exception:
            pass
            
    return np.array(best_coords).reshape(-1, 2)
