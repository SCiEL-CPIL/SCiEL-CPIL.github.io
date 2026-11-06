import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0

    # Configuration 1: 7,7,7,7,7,7,7,4
    rows1 = [7, 7, 7, 7, 7, 7, 7, 4]
    
    # Configuration 2: 8,7,7,7,7,7,7 (Total 50)
    rows2 = [8, 7, 7, 7, 7, 7, 7]

    # Configuration 3: 7,8,7,7,7,7,7 (Total 50) - Symmetric to 2 but middle row wider?
    # Let's just use rows1 and rows2.
    
    configs = [rows1, rows2]
    
    best_score = -1.0
    best_centers = None
    best_params = None
    best_rows = None

    for rows in configs:
        if time.time() > deadline - 10:
            break
            
        def get_centers(params, rows):
            dx, dy, shift, X, Y = params
            centers = []
            for i, n in enumerate(rows):
                y = Y + i * dy
                x_start = X - (n - 1) * dx / 2.0
                if i % 2 == 1:
                    x_start += shift * dx
                for j in range(n):
                    x = x_start + j * dx
                    centers.append([x, y])
            return np.array(centers)

        def score(params, rows):
            if time.time() > deadline:
                return 0.0
            centers = get_centers(params, rows)
            if not np.all(np.isfinite(centers)):
                return 0.0
            
            # Boundaries
            min_dist_boundary = np.min([
                np.min(0.5 - centers[:, 0]),
                np.min(0.5 + centers[:, 0]),
                np.min(0.5 - centers[:, 1]),
                np.min(0.5 + centers[:, 1])
            ])
            
            if min_dist_boundary <= 0:
                return 0.0
                
            # Pairs
            dists = pdist(centers)
            min_dist_pair = np.min(dists)
            
            r = min(min_dist_boundary, min_dist_pair / 2.0)
            
            if r <= 0:
                return 0.0
            return -r

        # Initial guess
        # Estimate dx, dy from area
        # Area = 1. N=50.
        # Hexagonal packing density ~ 0.9.
        # Area per circle ~ 1/50 = 0.02.
        # r ~ sqrt(0.02 / (pi * 0.9)) ~ 0.084? 
        # Wait, r=0.071 is known.
        # dx ~ 2*r = 0.142
        # dy ~ 2*r*sqrt(3)/2 = 0.123
        dx_init = 0.14
        dy_init = 0.12
        shift_init = 0.5
        X_init = 0.0
        Y_init = 0.0
        
        x0 = [dx_init, dy_init, shift_init, X_init, Y_init]
        
        res = minimize(
            score, 
            x0, 
            args=(rows,),
            method='Nelder-Mead', 
            options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 5000, 'maxfev': 5000}
        )
        
        if res.fun < 0 and -res.fun > best_score:
            best_score = -res.fun
            best_params = res.x
            best_rows = rows

    if best_centers is None and best_params is not None:
        best_centers = get_centers(best_params, best_rows)
        
    # If optimization failed or didn't find anything better than 0, fallback
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
