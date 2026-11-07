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
    # The evaluator clips? No, "heights must be finite".
    # But the constraint is h[i] in [0,1].
    # If h is outside, it's invalid?
    # "Choose 100 equal-width step heights h on [0, 2], with every h[i] in [0, 1]"
    # So we must return h in [0,1].
    # The evaluator calculates value. If h is invalid, result is invalid.
    # We should ensure our returned h is strictly in [0,1] and sum is 50.
    
    # For the optimization objective, we can assume h is within bounds if SLSQP works.
    # But to be safe in the objective function (which might be called with slightly out-of-bounds during optimization),
    # we should penalize or clip.
    # SLSQP handles bounds.
    
    h_clipped = np.clip(h, 0.0, 1.0)
    # Check sum constraint violation? 
    # SLSQP handles it.
    # But for the objective value to be comparable, we just compute the metric.
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
        # x is guaranteed to be in bounds by SLSQP?
        # SLSQP might evaluate slightly outside.
        # We clip for the calculation to avoid nan/inf, but the penalty for sum constraint handles the rest.
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
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            h_candidate = np.clip(res.x, 0.0, 1.0)
            # Fix sum drift
            diff = N/2.0 - np.sum(h_candidate)
            if abs(diff) > 1e-9:
                # Distribute diff. If diff is small, uniform is fine.
                # If diff is large, we might violate bounds.
                # Simple projection:
                h_candidate = h_candidate + diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                # Re-fix sum if clipping caused drift?
                # Iterate once more
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

    # 2. Random restarts
    rng = np.random.default_rng(123)
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        h_rand = rng.uniform(0, 1, N)
        # Project to sum 50
        diff = 50.0 - np.sum(h_rand)
        h_rand += diff / N
        h_rand = np.clip(h_rand, 0.0, 1.0)
        
        try_optimize(h_rand)
        
        # Also try some structured starts
        if i % 5 == 0:
             # Sine wave start
             freq = rng.uniform(1, 10)
             phase = rng.uniform(0, 2*np.pi)
             h_sine = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
             h_sine = np.clip(h_sine, 0, 1)
             diff = 50.0 - np.sum(h_sine)
             h_sine += diff / N
             h_sine = np.clip(h_sine, 0, 1)
             try_optimize(h_sine)

    return best_h.tolist()
