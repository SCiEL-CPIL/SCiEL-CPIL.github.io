import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_correlations(h):
    """Compute C_k for all k in full correlation mode."""
    return np.correlate(h, 1.0 - h, mode="full")

def _smooth_objective(h, tau):
    corr = _compute_correlations(h)
    max_corr = np.max(corr)
    if max_corr > 1e6: return 1e6
    # Log sum exp
    lse = np.log(np.sum(np.exp((corr - max_corr) / tau))) + max_corr / tau
    return tau * lse

def _exact_objective(h):
    corr = _compute_correlations(h)
    return np.max(corr)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_h = np.full(N, 0.5)
    best_val = _exact_objective(best_h)
    
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
    
    def try_update(h_cand):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h_cand)):
            return
        h_cand = np.clip(h_cand, 0.0, 1.0)
        diff = N/2.0 - np.sum(h_cand)
        if abs(diff) > 1e-9:
            h_cand += diff / N
            h_cand = np.clip(h_cand, 0.0, 1.0)
            
        val = _exact_objective(h_cand)
        if val < best_val:
            best_val = val
            best_h = h_cand.copy()

    # Generate diverse initializations
    np.random.seed(42)
    inits = [
        np.full(N, 0.5),
        np.linspace(0, 1, N),
        np.linspace(1, 0, N),
        np.random.uniform(0, 1, N),
        np.where(np.arange(N) % 2 == 0, 0.8, 0.2),
        np.where(np.arange(N) % 2 == 1, 0.8, 0.2),
        np.tile([0.0, 0.5, 1.0, 0.5], 25),
        np.tile([0.0, 0.0, 1.0, 1.0, 0.5], 20),
    ]
    
    # Sinusoidal initializations
    for f in [1, 2, 3, 4, 5]:
        h_s = 0.5 + 0.4 * np.cos(2 * np.pi * f * np.arange(N) / N)
        inits.append(h_s)
        
    # Block patterns
    h_block = np.zeros(N)
    h_block[:N//2] = 1.0
    inits.append(h_block)
    
    h_three = np.zeros(N)
    h_three[:N//3] = 1.0
    h_three[2*N//3:] = 1.0
    inits.append(h_three)

    # Normalize initializations to satisfy constraints approximately
    for i in range(len(inits)):
        h_init = np.clip(inits[i], 0, 1)
        diff = N/2.0 - np.sum(h_init)
        h_init += diff/N
        h_init = np.clip(h_init, 0, 1)
        inits[i] = h_init

    # Optimization Loop
    for h_init in inits:
        if time.time() > deadline - 10:
            break
            
        # Phase 1: Smooth optimization with SLSQP
        # Using a moderate tau to smooth the max function
        best_smooth_h = h_init.copy()
        
        for tau in [5.0, 1.0, 0.5]:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    lambda x: _smooth_objective(x, tau), 
                    best_smooth_h, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                if res.success and res.fun < _smooth_objective(best_smooth_h, tau):
                    best_smooth_h = res.x
            except:
                pass
        
        try_update(best_smooth_h)
        
        # Phase 2: Exact local refinement
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    _exact_objective, 
                    best_smooth_h, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                try_update(res.x)
            except:
                pass

    # Final global polish on the best found so far
    if time.time() < deadline - 2:
        try:
            res = minimize(
                _exact_objective, 
                best_h, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-15}
            )
            try_update(res.x)
        except:
            pass

    # Final constraint enforcement
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N/2.0 - np.sum(best_h)
    if abs(diff) > 1e-9:
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)

    return best_h
