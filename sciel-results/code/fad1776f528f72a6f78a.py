import numpy as np
import time
from scipy.optimize import minimize

N = 400


def compute_value(h):
    S = np.sum(h)
    if abs(S) < 1e-10:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (S * S))


def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    best_h = np.ones(N)
    best_val = 2.0

    def try_candidate(h):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        v = compute_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Define objective for optimization
    # We minimize the soft-max of the autoconvolution, with a penalty for sum != 1
    def objective(x):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(x, x)
        # Soft max approximation
        beta = 10.0
        m = np.max(conv)
        # Ensure numerical stability
        exp_conv = np.exp((conv - m) * beta)
        soft_max = m + np.log(np.sum(exp_conv)) / beta
        # Penalty for constraint sum(x) = 1
        penalty = 100.0 * (s - 1.0) ** 2
        return soft_max + penalty

    np.random.seed(42)

    # Phase 1: Multiple random restarts with L-BFGS-B
    for i in range(15):
        if time.time() > deadline - 20:
            break
        
        # Random start, normalized
        h0 = np.random.randn(N)
        h0 = h0 / np.sum(h0)
        
        try:
            res = minimize(objective, h0, method='L-BFGS-B', 
                           options={'maxiter': 50, 'maxfun': 200})
            h_new = res.x
            try_candidate(h_new)
        except Exception:
            pass

    # Phase 2: Refine the best solution found
    if time.time() < deadline - 10:
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) > 1e-10:
            h0 = h0 / s0
        
        try:
            res = minimize(objective, h0, method='L-BFGS-B',
                           options={'maxiter': 100, 'maxfun': 500})
            h_new = res.x
            try_candidate(h_new)
        except Exception:
            pass

    return best_h
