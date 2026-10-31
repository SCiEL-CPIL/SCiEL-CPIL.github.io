import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    # Seed result for fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    seed_result = np.array(cells[:N])
    seed_sum = np.sum(seed_result[:, 2])
    
    # Initial layout: 5x5 grid, skip 4 corners to get 21 circles
    # This provides a good starting point for local optimization
    # 5x5 grid coordinates (0.1, 0.3, 0.5, 0.7, 0.9)
    coords = [0.1, 0.3, 0.5, 0.7, 0.9]
    init_points = []
    for i in range(5):
        for j in range(5):
            # Skip corners (0,0), (0,4), (4,0), (4,4)
            if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                continue
            init_points.append((coords[i], coords[j]))
            
    # We have 21 points
    x0 = np.array([p[0] for p in init_points] + [p[1] for p in init_points])
    
    def objective(params):
        xs = params[:N]
        ys = params[N:]
        
        # Wall clearances
        wall_clears = np.concatenate([xs, 1-xs, ys, 1-ys])
        min_wall = np.min(wall_clears)
        
        # Pairwise clearances
        min_pair = np.inf
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_pair:
                    min_pair = dist
        min_pair_half = min_pair / 2.0
        
        min_clear = min(min_wall, min_pair_half)
        return -min_clear

    # Optimize
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7})
        
        best_params = res.x
        xs = best_params[:N]
        ys = best_params[N:]
        
        # Calculate the max uniform radius
        r = -res.fun
        
        # Ensure r is positive and reasonable
        if r > 0.05:
            # Check if sum is better
            new_sum = N * r
            if new_sum > seed_sum:
                result = np.zeros((N, 3))
                result[:, 0] = xs
                result[:, 1] = ys
                result[:, 2] = r
                return result
    except Exception:
        pass
        
    return seed_result
