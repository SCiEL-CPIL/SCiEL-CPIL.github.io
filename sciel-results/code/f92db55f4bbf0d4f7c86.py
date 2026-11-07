from scipy.optimize import minimize
import numpy as np
import time

N = 100
HALF = N // 2

def _compute_correlations(h):
    return np.correlate(h, 1.0 - h, mode="full")

def _exact_objective(h):
    corr = _compute_correlations(h)
    return np.max(corr)

def _binary_max_corr(h_bin):
    """Max correlation for binary h using vectorized correlate."""
    corr = np.correlate(h_bin, 1.0 - h_bin, mode="full")
    return np.max(corr)

def _binary_hill_climb(h_bin, max_iters=400, time_limit=3.0, start_time=None):
    """Hill climbing on binary vectors with exactly HALF ones."""
    if start_time is None:
        start_time = time.time()
    
    h = h_bin.copy().astype(float)
    current_max = _binary_max_corr(h)
    ones = np.where(h == 1)[0]
    zeros = np.where(h == 0)[0]
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        best_swap = None
        best_val = current_max
        n_ones = len(ones)
        n_zeros = len(zeros)
        n_tries = min(80, n_ones * n_zeros)
        
        for _ in range(n_tries):
            i = np.random.randint(n_ones)
            j = np.random.randint(n_zeros)
            op, zp = ones[i], zeros[j]
            h[op] = 0
            h[zp] = 1
            val = _binary_max_corr(h)
            h[op] = 1
            h[zp] = 0
            if val < best_val:
                best_val = val
                best_swap = (op, zp)
        
        if best_swap is None:
            # Random jump to escape local optimum
            for _ in range(2):
                i = np.random.randint(n_ones)
                j = np.random.randint(n_zeros)
                ones[i], zeros[j] = zeros[j], ones[i]
                h[ones[i]] = 1
                h[zeros[j]] = 0
            current_max = _binary_max_corr(h)
            ones = np.where(h == 1)[0]
            zeros = np.where(h == 0)[0]
            continue
        
        op, zp = best_swap
        h[op] = 0
        h[zp] = 1
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        current_max = best_val
    
    return h, current_max

def _project_simplex(y, target_sum):
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(y - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    h_final = np.clip(y - (lo + hi) / 2.0, 0.0, 1.0)
    diff = target_sum - np.sum(h_final)
    if abs(diff) > 1e-9:
        h_final += diff / N
        h_final = np.clip(h_final, 0.0, 1.0)
    return h_final

def _smooth_objective(h, tau):
    corr = _compute_correlations(h)
    max_corr = np.max(corr)
    if max_corr > 1e6:
        return 1e6
    lse = np.log(np.sum(np.exp((corr - max_corr) / tau))) + max_corr / tau
    return tau * lse

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_h = np.full(N, 0.5)
    best_val = _exact_objective(best_h)
    
    def try_update(h_cand):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h_cand)):
            return
        h_cand = np.clip(h_cand, 0.0, 1.0)
        diff = HALF - np.sum(h_cand)
        if abs(diff) > 1e-9:
            h_cand += diff / N
            h_cand = np.clip(h_cand, 0.0, 1.0)
        val = _exact_objective(h_cand)
        if val < best_val:
            best_val = val
            best_h = h_cand.copy()
    
    np.random.seed(42)
    
    # Phase 1: Binary hill climbing from random starts
    n_starts = 15
    for s in range(n_starts):
        if time.time() > deadline - 25:
            break
        
        h_bin = np.zeros(N)
        h_bin[np.random.choice(N, HALF, replace=False)] = 1
        
        h_opt, val = _binary_hill_climb(h_bin, max_iters=300, time_limit=2.5,
                                          start_time=time.time())
        h_cont = h_opt.copy()
        
        if time.time() < deadline - 5:
            try:
                res = minimize(_exact_objective, h_cont, method='SLSQP',
                              bounds=[(0.0, 1.0)] * N,
                              constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - HALF}],
                              options={'maxiter': 100, 'ftol': 1e-12})
                try_update(res.x)
            except:
                pass
        
        if time.time() < deadline - 5:
            try:
                for tau in [1.0, 0.3]:
                    res = minimize(lambda x: _smooth_objective(x, tau), h_cont,
                                  method='SLSQP', bounds=[(0.0, 1.0)] * N,
                                  constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - HALF}],
                                  options={'maxiter': 80, 'ftol': 1e-12})
                    h_cont = res.x
                    try_update(h_cont)
            except:
                pass
    
    # Phase 2: Continuous optimization from diverse initializations
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
    for f in [1, 2, 3]:
        inits.append(0.5 + 0.4 * np.cos(2 * np.pi * f * np.arange(N) / N))
    
    for h_init in inits:
        if time.time() > deadline - 8:
            break
        
        h = np.clip(h_init, 0, 1)
        diff = HALF - np.sum(h)
        h += diff / N
        h = np.clip(h, 0, 1)
        
        for tau in [3.0, 1.0, 0.3]:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(lambda x: _smooth_objective(x, tau), h, method='SLSQP',
                              bounds=[(0.0, 1.0)] * N,
                              constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - HALF}],
                              options={'maxiter': 80, 'ftol': 1e-12})
                h = res.x
                try_update(h)
            except:
                break
        
        if time.time() < deadline - 3:
            try:
                res = minimize(_exact_objective, h, method='SLSQP',
                              bounds=[(0.0, 1.0)] * N,
                              constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - HALF}],
                              options={'maxiter': 80, 'ftol': 1e-12})
                try_update(res.x)
            except:
                pass
    
    # Final polish
    if time.time() < deadline - 2:
        try:
            res = minimize(_exact_objective, best_h, method='SLSQP',
                          bounds=[(0.0, 1.0)] * N,
                          constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - HALF}],
                          options={'maxiter': 150, 'ftol': 1e-14})
            try_update(res.x)
        except:
            pass
    
    return best_h
