import numpy as np
from scipy.optimize import minimize
import time

N = 100
HALF = N / 2.0

def _value(h):
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,):
        return np.inf
    if np.any(~np.isfinite(h)):
        return np.inf
    if np.any(h < -1e-12) or np.any(h > 1 + 1e-12):
        return np.inf
    if abs(np.sum(h) - HALF) > 1e-9 * N:
        return np.inf
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr)) / N * 2.0

def _corr_max(h):
    """Fast computation of max correlation (no validity checks)."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr))

def _epigraph_objective(x):
    """Objective: minimize t (first element), ignore h (rest)."""
    return x[0]

def _make_epigraph_constraints():
    """Build constraint functions for epigraph formulation."""
    def sum_constraint(x):
        return np.sum(x[1:]) - HALF
    
    lags = np.arange(-(N-1), N)
    
    def make_corr_constraint(k_idx):
        k = lags[k_idx]
        def constraint(x):
            h = x[1:]
            t = x[0]
            if k >= 0:
                if k == 0:
                    c = np.sum(h * (1.0 - h))
                else:
                    c = np.sum(h[:N-k] * (1.0 - h[k:]))
            else:
                kk = -k
                c = np.sum(h[kk:] * (1.0 - h[:N-kk]))
            return t - c
        return constraint
    
    constraints = [{'type': 'eq', 'fun': sum_constraint, 'jac': lambda x: np.concatenate([np.zeros(1), np.ones(N)])}]
    for k_idx in range(2*N - 1):
        constraints.append({'type': 'ineq', 'fun': make_corr_constraint(k_idx)})
    
    return constraints

def _local_refine(h, t, maxiter=50):
    """Local refinement with tighter tolerance."""
    x0 = np.concatenate([[t], h])
    bounds = [(max(0.0, t - 0.5), t + 0.5)] + [(0.0, 1.0)] * N
    
    try:
        res = minimize(
            _epigraph_objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=_make_epigraph_constraints(),
            options={'maxiter': maxiter, 'ftol': 1e-14}
        )
        if res.success:
            h_new = np.clip(res.x[1:], 0.0, 1.0)
            t_new = res.x[0]
            diff = HALF - np.sum(h_new)
            if abs(diff) > 1e-10:
                h_new = h_new + diff / N
                h_new = np.clip(h_new, 0.0, 1.0)
            val = _value(h_new)
            if np.isfinite(val):
                return h_new, val
    except Exception:
        pass
    
    return h, _value(h)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0
    
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _value(best_h)
    
    constraints = _make_epigraph_constraints()
    bounds = [(0.0, 2.0)] + [(0.0, 1.0)] * N
    
    def try_optimize(start_h, maxiter=100):
        nonlocal best_h, best_val
        t_init = _corr_max(start_h)
        x0 = np.concatenate([[t_init], start_h])
        
        try:
            res = minimize(
                _epigraph_objective, x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            h_candidate = np.clip(res.x[1:], 0.0, 1.0)
            t_candidate = res.x[0]
            
            diff = HALF - np.sum(h_candidate)
            if abs(diff) > 1e-9:
                h_candidate = h_candidate + diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            h_refined, val_refined = _local_refine(h_candidate, t_candidate, maxiter=30)
            
            if val_refined < best_val:
                best_h = h_refined
                best_val = val_refined
        except Exception:
            pass

    # 1. Try seed
    try_optimize(h0, maxiter=200)
    
    if time.time() > deadline:
        return best_h.tolist()

    # 2. Random restarts
    rng = np.random.default_rng(42)
    num_restarts = 30
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        h_rand = rng.uniform(0, 1, N)
        diff = HALF - np.sum(h_rand)
        h_rand += diff / N
        h_rand = np.clip(h_rand, 0.0, 1.0)
        
        try_optimize(h_rand)
        
        if i % 3 == 0:
             freq = rng.uniform(1, 10)
             phase = rng.uniform(0, 2*np.pi)
             h_sine = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
             h_sine = np.clip(h_sine, 0, 1)
             diff = HALF - np.sum(h_sine)
             h_sine += diff / N
             h_sine = np.clip(h_sine, 0, 1)
             try_optimize(h_sine)

    return best_h.tolist()
