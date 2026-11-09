import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0

    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.01, 1.0)) # s

    def objective(params):
        if time.time() > deadline:
            return float('inf')
        x = params[0::3]
        y = params[1::3]
        s = params[2::3]
        
        pen_boundary = 0.0
        for i in range(n):
            pen_boundary += max(0.0, s[i]/2.0 - x[i])
            pen_boundary += max(0.0, x[i] - (1.0 - s[i]/2.0))
            pen_boundary += max(0.0, s[i]/2.0 - y[i])
            pen_boundary += max(0.0, y[i] - (1.0 - s[i]/2.0))
        
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                w = max(0.0, (s[i] + s[j]) / 2.0 - abs(x[i] - x[j]))
                h = max(0.0, (s[i] + s[j]) / 2.0 - abs(y[i] - y[j]))
                pen_overlap += w * h
                
        return -np.sum(s) + 100.0 * (pen_boundary + pen_overlap)

    # Initial guess
    k = 3
    side = 1.0 / k
    x0 = np.zeros(n)
    y0 = np.zeros(n)
    s0 = np.zeros(n)
    for i in range(n):
        col = i % k
        row = i // k
        x0[i] = (col + 0.5) * side
        y0[i] = (row + 0.5) * side
        s0[i] = side
        
    x0_params = np.zeros(3 * n)
    x0_params[0::3] = x0
    x0_params[1::3] = y0
    x0_params[2::3] = s0

    best_params = x0_params.copy()
    best_score = objective(best_params)

    # Global search
    if time.time() < deadline - 30.0:
        try:
            de_result = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=2000, 
                popsize=40, 
                tol=1e-8,
                polish=False
            )
            if de_result.fun < best_score:
                best_params = de_result.x
                best_score = de_result.fun
        except Exception:
            pass

    # Local refinement
    if time.time() < deadline - 10.0:
        try:
            res = minimize(
                objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass

    x = best_params[0::3]
    y = best_params[1::3]
    s = best_params[2::3]
    
    centers = np.stack([x, y], axis=1)
    sides = s
    
    return centers, sides
