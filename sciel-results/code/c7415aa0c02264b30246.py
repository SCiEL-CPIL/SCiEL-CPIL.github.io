"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv)) / (s ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = None
    best_value = float('inf')
    
    def update_best(h):
        nonlocal best_h, best_value
        if h is None:
            return
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-12:
            return
        val = _compute_value(h)
        if val < best_value:
            best_value = val
            best_h = h.copy()
    
    # Strategy: The value is scale-invariant. We want to minimize max(conv(h,h)) 
    # when sum(h) is fixed. Key insight: if we can make all autoconvolution 
    # coefficients negative, max(conv) is the least negative, and we want it 
    # close to 0.
    
    # Try alternating patterns with small bias to make sum non-zero
    # h_i = (-1)^i * a + b, where b is small bias
    # sum = b*N (since alternating sum is 0 for even N)
    # We can optimize a and b
    
    # First, try simple heuristic: alternating with small perturbation
    for bias in [0.001, 0.01, 0.1, 1.0]:
        h = np.array([(-1)**i for i in range(N)], dtype=float)
        h += bias
        update_best(h)
    
    # Try: h_i = (-1)^i + c, optimize c
    def opt_alt_bias(c):
        h = np.array([(-1)**i for i in range(N)], dtype=float) + c
        return _compute_value(h)
    
    # Grid search for bias
    for c in np.linspace(0.001, 5.0, 200):
        if time.time() > deadline:
            break
        h = np.array([(-1)**i for i in range(N)], dtype=float) + c
        update_best(h)
    
    # Now try optimization from good starting points
    # Use the best found so far or alternating with small bias
    if best_h is not None:
        h0 = best_h.copy()
    else:
        h0 = np.array([(-1)**i for i in range(N)], dtype=float) + 0.1
    
    # Normalize h0 to have sum = 1 for optimization
    s0 = np.sum(h0)
    if abs(s0) > 1e-9:
        h0 = h0 / s0
    
    # Objective: minimize |2N * max(conv(h,h))| / sum(h)^2
    # Since scale invariant, equivalent to minimizing max(conv(h,h)) when sum(h)=1
    # But sum(h) may not stay 1 during optimization, so use full expression
    
    def objective(h):
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        conv = np.convolve(h, h)
        return abs(2 * N * np.max(conv)) / (s ** 2)
    
    # Try L-BFGS-B optimization
    try:
        res = minimize(objective, h0, method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        update_best(res.x)
    except Exception:
        pass
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization with some structure
        # Mix of alternating and random
        base = np.array([(-1)**i for i in range(N)], dtype=float)
        noise = rng.standard_normal(N) * 0.5
        h_rand = base + noise
        update_best(h_rand)
        
        # Optimize from this point
        s = np.sum(h_rand)
        if abs(s) < 1e-6:
            continue
        h_opt = h_rand / s
        
        try:
            res = minimize(objective, h_opt, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12})
            update_best(res.x)
        except Exception:
            pass
    
    # Try pure random search
    n_random = 50
    for i in range(n_random):
        if time.time() > deadline:
            break
        h_r = rng.standard_normal(N)
        update_best(h_r)
    
    # Try structured patterns: blocks of +1 and -1
    # e.g., first k elements +1, rest -1, or alternating blocks
    for k in range(1, N):
        if time.time() > deadline:
            break
        h_block = np.ones(N)
        h_block[k:] = -1
        update_best(h_block)
    
    # Try: h_i = 1 if i mod 2 == 0 else -1, but with varying period
    for period in [2, 3, 4, 5, 7, 10]:
        if time.time() > deadline:
            break
        h_periodic = np.array([1.0 if (i % period) < period//2 else -1.0 for i in range(N)])
        update_best(h_periodic)
        # Add small bias
        for bias in [0.01, 0.1, 1.0]:
            h_b = h_periodic + bias
            update_best(h_b)
    
    # Final: if we have a good candidate, try local refinement
    if best_h is not None:
        h_refine = best_h.copy()
        s = np.sum(h_refine)
        if abs(s) > 1e-9:
            h_refine = h_refine / s
            try:
                res = minimize(objective, h_refine, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-14})
                update_best(res.x)
            except Exception:
                pass
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
