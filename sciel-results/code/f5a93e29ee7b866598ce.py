import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _value(h):
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,):
        return np.inf
    if np.any(~np.isfinite(h)):
        return np.inf
    h_clipped = np.clip(h, 0.0, 1.0)
    corr = np.correlate(h_clipped, 1.0 - h_clipped, mode="full")
    return float(np.max(corr)) / N * 2.0

def construct():
    t_start = time.time()
    deadline = t_start + 110.0
    
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _value(best_h)
    
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
    
    def objective(x):
        h = np.clip(x, 0.0, 1.0)
        corr = np.correlate(h, 1.0 - h, mode="full")
        return np.max(corr)
        
    def try_optimize(start_h):
        nonlocal best_h, best_val
        try:
            res = minimize(
                objective, start_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            h_candidate = np.clip(res.x, 0.0, 1.0)
            # Fix sum drift
            diff = N/2.0 - np.sum(h_candidate)
            if abs(diff) > 1e-9:
                h_candidate = h_candidate + diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                diff = N/2.0 - np.sum(h_candidate)
                if abs(diff) > 1e-9:
                    h_candidate = h_candidate + diff / N
                    h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            val = _value(h_candidate)
            if val < best_val:
                best_h = h_candidate
                best_val = val
        except Exception:
            pass

    # 1. Try seed
    try_optimize(h0)
    
    if time.time() > deadline:
        return best_h.tolist()

    # 2. Structured and Random restarts
    rng = np.random.default_rng(123)
    
    # Generate a list of starting points
    starts = []
    
    # Sine waves with various frequencies and phases
    for freq in range(1, 20):
        for phase in np.linspace(0, 2*np.pi, 5, endpoint=False):
            h_sine = 0.5 + 0.45 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            h_sine = np.clip(h_sine, 0, 1)
            diff = 50.0 - np.sum(h_sine)
            h_sine += diff / N
            h_sine = np.clip(h_sine, 0, 1)
            starts.append(h_sine)
            
    # Random
    for _ in range(30):
        h_rand = rng.uniform(0, 1, N)
        diff = 50.0 - np.sum(h_rand)
        h_rand += diff / N
        h_rand = np.clip(h_rand, 0, 1)
        starts.append(h_rand)
        
    # Multi-frequency
    for _ in range(20):
        h_multi = np.full(N, 0.5)
        for _ in range(3):
            f = rng.integers(1, 15)
            p = rng.uniform(0, 2*np.pi)
            a = rng.uniform(0.1, 0.3)
            h_multi += a * np.sin(2 * np.pi * f * np.arange(N) / N + p)
        h_multi = np.clip(h_multi, 0, 1)
        diff = 50.0 - np.sum(h_multi)
        h_multi += diff / N
        h_multi = np.clip(h_multi, 0, 1)
        starts.append(h_multi)

    for i, start_h in enumerate(starts):
        if time.time() > deadline - 5:
            break
        try_optimize(start_h)

    return best_h.tolist()
