"""Rewritten program for the second autocorrelation inequality task."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
W = 0.5 / (N + 1)  # spacing w for the grid of length m+2 where m=N+1=51

def compute_value(h):
    """
    h: array of 50 nonnegative step heights.
    Returns the value l2 / (norm_1 * norm_inf).
    """
    h = np.asarray(h, dtype=float)
    if h.shape != (N,):
        h = h.reshape(N)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    
    g = np.convolve(h, h)  # length 2N-1 = 99
    m = len(g)  # 99
    # y has length m+2 = 101
    y = np.zeros(m + 2)
    y[1:m+1] = g
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # i ranges from 0 to m inclusive, so there are m+1 terms
    # y[i] and y[i+1] for i=0..m
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (W / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    return l2 / (norm_1 * norm_inf)

def objective(h):
    """Minimize negative of value for scipy.optimize."""
    h = np.clip(h, 0, None)
    return -compute_value(h)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Strategy 1: Differential Evolution for global exploration
    try:
        if time.time() < deadline:
            result_de = differential_evolution(
                objective,
                bounds=[(0, 10)] * N,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                callback=lambda x, convergence: None
            )
            if result_de.success or result_de.fun < -best_val:
                cand_h = np.clip(result_de.x, 0, None)
                cand_val = compute_value(cand_h)
                if cand_val > best_val:
                    best_h = cand_h
                    best_val = cand_val
    except Exception:
        pass
    
    # Strategy 2: Nelder-Mead polish from best found
    try:
        if time.time() < deadline:
            result_nm = minimize(
                objective,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            cand_h = np.clip(result_nm.x, 0, None)
            cand_val = compute_value(cand_h)
            if cand_val > best_val:
                best_h = cand_h
                best_val = cand_val
    except Exception:
        pass
    
    # Strategy 3: Try diverse initial patterns and polish
    diverse_starts = []
    
    # Power law
    x = np.linspace(0, 1, N)
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        diverse_starts.append(x ** alpha)
    
    # Sparse spikes
    for k in [1, 2, 3, 5]:
        for pos in range(0, N, max(1, N // k)):
            h_spike = np.zeros(N)
            h_spike[pos] = 1.0
            diverse_starts.append(h_spike)
    
    # Bump functions
    for center in [N // 4, N // 2, 3 * N // 4]:
        for width in [2, 3, 5]:
            h_bump = np.zeros(N)
            for i in range(N):
                if abs(i - center) <= width:
                    h_bump[i] = np.exp(-((i - center) ** 2) / (2 * width ** 2))
            diverse_starts.append(h_bump)
    
    # Random starts
    np.random.seed(123)
    for _ in range(20):
        diverse_starts.append(np.random.rand(N))
    
    for start_h in diverse_starts:
        if time.time() > deadline - 5.0:
            break
        try:
            start_h = np.asarray(start_h, dtype=float).copy()
            start_h = np.clip(start_h, 0, None)
            
            # Quick Nelder-Mead
            result = minimize(
                objective,
                start_h,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            cand_h = np.clip(result.x, 0, None)
            cand_val = compute_value(cand_h)
            if cand_val > best_val:
                best_h = cand_h
                best_val = cand_val
        except Exception:
            continue
    
    # Final polish on the best
    try:
        if time.time() < deadline - 2.0:
            result_final = minimize(
                objective,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            cand_h = np.clip(result_final.x, 0, None)
            cand_val = compute_value(cand_h)
            if cand_val > best_val:
                best_h = cand_h
                best_val = cand_val
    except Exception:
        pass
    
    # Ensure nonnegativity and finiteness
    best_h = np.clip(best_h, 0, None)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    
    return list(best_h)

# Keep the original function name
construct = construct
