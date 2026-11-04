import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    # We optimize the angular gaps between consecutive points on a circle.
    # There are N gaps, but they sum to 2*pi, so we have N-1 free variables.
    
    def objective(x):
        # x: array of first N-1 gaps
        # Check for invalid gaps (must be positive)
        if np.any(x <= 0):
            return 0.0
            
        deltas = np.empty(N)
        deltas[:N-1] = x
        deltas[N-1] = 2 * np.pi - np.sum(x)
        
        # Check if the last gap is valid
        if deltas[N-1] <= 0:
            return 0.0
            
        # Calculate cumulative angles (starting at 0)
        # phi[i] is the angle of point i
        phi = np.zeros(N)
        phi[1:] = np.cumsum(deltas[:N-1])
        
        # Calculate the minimum triangle area numerator
        # Vectorized calculation of all triangle areas
        # i, j, k indices
        # We can use broadcasting to compute all combinations
        # But N is small, so a simple loop might be faster than array overhead if not careful.
        # Let's stick to the loop but optimize it slightly or just rely on restarts.
        
        min_num = float('inf')
        
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = phi[j] - phi[i]
                    b = phi[k] - phi[j]
                    c = 2 * np.pi - a - b
                    val = np.sin(a) + np.sin(b) + np.sin(c)
                    if val < min_num:
                        min_num = val
        
        den = np.sum(np.sin(deltas))
        
        if den <= 0:
            return 0.0
            
        ratio = min_num / den
        return -ratio

    bounds = [(1e-5, 2 * np.pi - 1e-5)] * (N-1)
    
    best_ratio = -1
    best_x = None
    
    # Run multiple restarts
    for _ in range(5):
        if best_x is None:
            # First run: uniform
            delta_init = np.full(N-1, 2 * np.pi / N)
        else:
            # Subsequent runs: perturb the best found so far
            delta_init = best_x + np.random.normal(0, 0.1, N-1)
            # Ensure positive and sum constraint roughly (will be corrected by optimizer)
            # Clip to bounds roughly to help optimizer
            delta_init = np.clip(delta_init, 1e-5, 2 * np.pi - 1e-5)
            # Normalize to sum to 2*pi? No, the last gap handles the sum.
            # But if the sum of first N-1 is too large, the last gap becomes negative.
            # We should scale them to ensure the last gap is positive.
            s = np.sum(delta_init)
            if s >= 2 * np.pi:
                 # Scale down
                 scale = (2 * np.pi - 1e-5) / s
                 delta_init = delta_init * scale
            
        result = minimize(
            objective, 
            delta_init, 
            method='Nelder-Mead', 
            bounds=bounds,
            options={
                'xatol': 1e-8, 
                'fatol': 1e-8, 
                'maxiter': 1000, 
                'adaptive': True
            }
        )
        
        if result.success:
            opt_deltas = np.empty(N)
            opt_deltas[:N-1] = result.x
            opt_deltas[N-1] = 2 * np.pi - np.sum(result.x)
            
            if np.all(opt_deltas > 0):
                curr_ratio = -result.fun
                if curr_ratio > best_ratio:
                    best_ratio = curr_ratio
                    best_x = result.x.copy()
                    best_deltas = opt_deltas.copy()
    
    if best_x is None:
        opt_deltas = np.full(N, 2 * np.pi / N)
    else:
        opt_deltas = best_deltas
        
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
