import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Objective function with penalty for sum constraint
    def objective(h):
        h_clipped = np.clip(h, 0, 1)
        # Calculate overlaps
        overlaps = np.array([np.sum(h_clipped * (1 - np.roll(h_clipped, t))) for t in range(N)])
        # Smooth max approximation
        beta = 50.0
        m = np.max(overlaps)
        smooth_max = m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta
        
        # Penalty for sum constraint
        sum_penalty = 1000.0 * (np.sum(h_clipped) - N / 2.0) ** 2
        
        return smooth_max + sum_penalty

    # Initial guess: 0.5 + small sinusoidal perturbation
    x0 = np.full(N, 0.5)
    x0 += 0.1 * np.sin(2 * np.pi * np.arange(N) / N)
    x0 = np.clip(x0, 0, 1)
    
    bounds = [(0.0, 1.0)] * N
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
        h = np.clip(res.x, 0, 1)
    except Exception:
        h = np.full(N, 0.5)
        
    # Ensure sum is exactly N/2
    s = np.sum(h)
    if s > 0:
        h = h * (N / 2.0) / s
    h = np.clip(h, 0, 1)
    
    # Fine-tune sum to be exactly N/2
    diff = N / 2.0 - np.sum(h)
    # Adjust the largest elements to minimize clipping issues
    idx = np.argsort(h)[::-1]
    for i in idx:
        if diff > 0:
            add = min(diff, 1.0 - h[i])
            h[i] += add
            diff -= add
        elif diff < 0:
            sub = min(-diff, h[i])
            h[i] -= sub
            diff += sub
        else:
            break
            
    return h
