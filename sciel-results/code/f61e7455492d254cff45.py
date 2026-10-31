"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _autoconv(h):
    """Compute autoconvolution of h."""
    n = len(h)
    c = np.zeros(n)
    for k in range(n):
        c[k] = np.dot(h[:n-k], h[k:])
    return c

def _score(h):
    """Compute the ratio: (L2 norm)^2 / (L1 norm * sup autoconv)."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    c = _autoconv(h)
    sup_c = np.max(c)
    if sup_c <= 0:
        return 0.0
    return l2_sq / (l1 * sup_c)

def _negative_score(h):
    return -_score(h)

def construct():
    deadline = time.time() + 110  # safety margin
    
    # Start with uniform
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try a few different initializations and optimize
    np.random.seed(42)
    
    # Strategy 1: Coordinate descent / local search
    # Try perturbing and improving
    h = best_h.copy()
    
    # Use scipy minimize with bounds
    bounds = [(0, None)] * N
    
    # Try multiple starting points
    starts = []
    starts.append(np.ones(N))
    
    # Random nonnegative starts
    for _ in range(3):
        starts.append(np.random.exponential(1.0, size=N))
    
    # A "bump" profile
    t = np.linspace(0, 1, N)
    starts.append(np.cos(np.pi * t)**2 + 0.1)
    
    # Try optimizing from each start
    for start in starts:
        if time.time() > deadline:
            break
        
        h0 = start.copy()
        # Normalize to avoid trivial scaling issues
        if np.sum(h0) > 0:
            h0 = h0 / np.sum(h0) * N  # keep L1 ~ N
        
        res = minimize(_negative_score, h0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if res.success or res.fun < _negative_score(h0):
            h_candidate = np.maximum(res.x, 0)
            s = _score(h_candidate)
            if s > best_score:
                best_score = s
                best_h = h_candidate
    
    # Local coordinate descent refinement
    h = best_h.copy()
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            old_val = h[i]
            # Try a few perturbation values
            candidates = [0.0, old_val * 0.5, old_val * 1.5, old_val + 0.1, old_val - 0.1, 1.0, 2.0]
            # Ensure nonnegative
            candidates = [max(0.0, c) for c in candidates]
            
            best_cand = old_val
            best_cand_score = _score(h)
            
            for c in candidates:
                if c == old_val:
                    continue
                h[i] = c
                s = _score(h)
                if s > best_cand_score + 1e-12:
                    best_cand_score = s
                    best_cand = c
            
            h[i] = best_cand
            if best_cand != old_val:
                improved = True
        
        if not improved:
            break
    
    final_score = _score(h)
    if final_score > best_score:
        best_h = h
        best_score = final_score
    
    return list(best_h)
# EVOLVE-BLOCK-END

