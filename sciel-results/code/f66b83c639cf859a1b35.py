import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 100

def construct():
    deadline = time.time() + 130  # Safety margin
    
    def max_overlap(h):
        # Circular shift: h_{i-1} for i=0..N-1
        h_prev = np.roll(h, 1)
        products = h * (1 - h_prev)
        return np.max(products)
    
    def objective(x):
        # x are the first N-1 variables, h[N-1] is determined by sum constraint
        h = np.empty(N)
        h[:N-1] = x
        h[N-1] = N/2.0 - np.sum(x)
        # Project to [0,1] approximately - use a penalty approach instead
        # Actually, let's use a smooth penalty for bounds
        penalty = 0.0
        for v in h:
            if v < 0:
                penalty += 100.0 * v * v
            elif v > 1:
                penalty += 100.0 * (v - 1) * (v - 1)
        if penalty > 1e-6:
            return max_overlap(h) + penalty
        return max_overlap(h)
    
    # Initial guess: alternating 0.5
    x0 = np.full(N-1, 0.5)
    
    # Try multiple initializations
    best_h = None
    best_score = np.inf
    
    # Initialization 1: uniform 0.5
    h_init = np.full(N, 0.5)
    x_init = h_init[:N-1].copy()
    res = minimize(objective, x_init, method='L-BFGS-B', 
                   options={'maxiter': 2000, 'ftol': 1e-12})
    h = np.empty(N)
    h[:N-1] = res.x
    h[N-1] = N/2.0 - np.sum(res.x)
    # Clamp to [0,1] and renormalize
    h = np.clip(h, 0, 1)
    h = h / np.sum(h) * (N/2.0)
    score = max_overlap(h)
    if score < best_score:
        best_score = score
        best_h = h.copy()
    
    if time.time() > deadline:
        return best_h if best_h is not None else np.full(N, 0.5)
    
    # Initialization 2: alternating pattern with slight perturbation
    for trial in range(5):
        if time.time() > deadline:
            break
        x_init = np.full(N-1, 0.5)
        x_init += 0.1 * np.random.randn(N-1)
        x_init = np.clip(x_init, 0.01, 0.99)
        # Adjust last element to satisfy sum
        x_init[N-2] = N/2.0 - np.sum(x_init[:N-2])
        if x_init[N-2] < 0 or x_init[N-2] > 1:
            continue
        
        res = minimize(objective, x_init, method='L-BFGS-B',
                       options={'maxiter': 2000, 'ftol': 1e-12})
        h = np.empty(N)
        h[:N-1] = res.x
        h[N-1] = N/2.0 - np.sum(res.x)
        h = np.clip(h, 0, 1)
        h = h / np.sum(h) * (N/2.0)
        score = max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    if time.time() > deadline:
        return best_h if best_h is not None else np.full(N, 0.5)
    
    # Try differential evolution for global search
    try:
        def de_objective(x):
            h = np.empty(N)
            h[:N-1] = x
            h[N-1] = N/2.0 - np.sum(x)
            # Check bounds
            if np.any(h < -0.01) or np.any(h > 1.01):
                return 10.0
            h = np.clip(h, 0, 1)
            h = h / np.sum(h) * (N/2.0)
            return max_overlap(h)
        
        bounds = [(0, 1)] * (N-1)
        de_res = differential_evolution(de_objective, bounds, maxiter=100, 
                                         tol=1e-8, seed=42, polish=True)
        h = np.empty(N)
        h[:N-1] = de_res.x
        h[N-1] = N/2.0 - np.sum(de_res.x)
        h = np.clip(h, 0, 1)
        h = h / np.sum(h) * (N/2.0)
        score = max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    except Exception:
        pass
    
    if best_h is None:
        return np.full(N, 0.5)
    
    # Final verification and cleanup
    best_h = np.clip(best_h, 0, 1)
    # Ensure exact sum
    best_h = best_h / np.sum(best_h) * (N/2.0)
    
    return best_h
