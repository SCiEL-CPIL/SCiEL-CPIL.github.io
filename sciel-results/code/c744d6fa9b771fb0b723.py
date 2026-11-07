import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 50
# The domain for h is indices 0..49 corresponding to [-1/4, 1/4]
# x_i = -0.25 + i * (0.5 / 49)
x_grid = np.linspace(-0.25, 0.25, N)

def _eval(h):
    """
    Computes the objective value for a given height vector h.
    h is assumed to be non-negative.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        h = np.clip(h, 0, None)
    if np.sum(h) == 0:
        return 0.0
        
    g = np.convolve(h, h, mode='full')
    m = len(g)
    
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    y_curr = y[:m+1]
    y_next = y[1:m+2]
    terms = y_curr**2 + y_curr * y_next + y_next**2
    S = np.sum(terms)
    l2 = (w / 3.0) * S
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0
        
    value = l2 / (norm_1 * norm_inf)
    return value

def _tent_to_h(params):
    """
    Convert tent basis parameters to heights h.
    params: [c1, h1, c2, h2, c3, h3, c4, h4, c5, h5]
    K=5 Tents.
    c: center (position), h: height (amplitude).
    Width is fixed to 0.25 (or scaled by height? No, let's keep width fixed or variable).
    Let's use variable width: [c1, h1, w1, c2, h2, w2, ...]
    Total params: 5 * 3 = 15.
    c in [-0.25, 0.25]
    h in [0, 10]
    w in [0.01, 0.5]
    """
    K = 5
    h = np.zeros(N)
    for i in range(K):
        c = params[i*3]
        amp = params[i*3 + 1]
        w = params[i*3 + 2]
        
        # Tent function: max(0, 1 - |x - c| / w)
        # Scale by amp
        # Vectorized operation
        x = x_grid
        tent = np.maximum(0, 1 - np.abs(x - c) / w)
        h += amp * tent
    
    # Enforce non-negativity (already positive)
    h = np.clip(h, 0, None)
    
    # Normalize to sum=1 for scale invariance stability
    s_sum = np.sum(h)
    if s_sum == 0:
        return np.ones(N) / N
    h = h / s_sum
    
    return h

def _objective(params):
    """
    Objective function for minimization (returns negative value).
    """
    try:
        h = _tent_to_h(params)
        val = _eval(h)
        return -val
    except Exception:
        return 0.0 # Large penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # Define bounds for the 15 parameters: [c1, h1, w1, c2, h2, w2, ...]
    # Centers: [-0.25, 0.25]
    # Heights: [0, 10]
    # Widths: [0.01, 0.5]
    
    bounds = []
    for _ in range(5):
        bounds.append((-0.25, 0.25)) # Center
        bounds.append((0.0, 10.0))   # Height
        bounds.append((0.01, 0.5))   # Width
    
    best_params = None
    best_val = -np.inf
    
    # Run Differential Evolution
    # 15 dimensions is small enough for DE to be effective and fast
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=300, 
            popsize=30, 
            tol=1e-8, 
            seed=42,
            polish=False,
            mutation=(0.5, 1.0),
            recombination=0.8
        )
        
        if res.fun < best_val:
            best_val = -res.fun
            best_params = res.x.copy()
            
    except Exception:
        pass
        
    # If time allows, try a few more restarts with different seeds
    if time.time() < deadline - 30:
        for seed in [123, 456, 789]:
            if time.time() > deadline - 15:
                break
            try:
                res2 = differential_evolution(
                    _objective, 
                    bounds, 
                    maxiter=150, 
                    popsize=20, 
                    tol=1e-8, 
                    seed=seed,
                    polish=False,
                    mutation=(0.5, 1.0),
                    recombination=0.8
                )
                if res2.fun < best_val:
                    best_val = -res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass

    # If we have significant time left, polish the best solution using L-BFGS-B
    if best_params is not None and time.time() < deadline - 10:
        try:
            res_polish = minimize(
                _objective, 
                best_params, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_polish.fun < best_val:
                best_val = -res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass

    # If DE failed completely, fallback to a simple heuristic
    if best_params is None:
        # Try a single central tent
        best_params = np.array([0.0, 1.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25])
        best_val = -_objective(best_params)

    # Convert best params to heights
    best_h = _tent_to_h(best_params)
    
    # Ensure non-negative and finite
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) == 0:
        best_h = np.ones(N) / N
        
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
