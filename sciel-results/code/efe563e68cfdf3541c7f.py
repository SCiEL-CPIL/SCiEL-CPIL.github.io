import numpy as np
from scipy.optimize import linprog
import time

N = 50

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    best_h = np.ones(N)
    best_score = 0.0
    
    def compute_score(h):
        h = np.asarray(h, dtype=float)
        if h.min() < -1e-12:
            return 0.0
        l1 = h.sum()
        if l1 < 1e-12:
            return 0.0
        autoconv = np.convolve(h, h, mode='full')
        peak = autoconv.max()
        if peak < 1e-12:
            return 0.0
        return (h @ h) / (l1 * peak)
    
    best_score = compute_score(best_h)
    
    # Strategy 1: Sidon set with exponential weights
    # A Sidon set in [0, N-1] where all pairwise sums are distinct
    # For small sets, we can use known Sidon sets
    # Let's try a few candidate Sidon sets
    
    def try_sidon_exp(support, base):
        nonlocal best_h, best_score
        h = np.zeros(N)
        n = len(support)
        for i, idx in enumerate(support):
            if 0 <= idx < N:
                h[idx] = base ** i
        score = compute_score(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Small Sidon sets (Bose-Chowla type)
    # {0, 1, 4, 10} - check: 0+1=1, 0+4=4, 0+10=10, 1+4=5, 1+10=11, 4+10=14 - all distinct
    sidon_sets = [
        [0, 1, 4, 10],
        [0, 1, 4, 14],
        [0, 1, 5, 14],
        [0, 2, 5, 13],
        [0, 1, 3, 9, 20],  # larger Sidon
        [0, 1, 4, 10, 23],
        [1, 2, 5, 13],  # not containing 0
        [2, 5, 11, 23],
    ]
    
    bases = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    
    for sset in sidon_sets:
        for b in bases:
            if time.time() > deadline:
                return best_h.tolist()
            try_sidon_exp(sset, b)
    
    # Strategy 2: LP-based approach
    # Maximize sum(h^2) / max(h*h) subject to sum(h)=1, h>=0
    # Use bisection on t: for given t, maximize sum(h^2) s.t. sum(h)=1, h>=0, h*h <= t
    # h*h <= t means for all k, sum_i h_i h_{k-i} <= t
    # This is a QP for fixed t, but we can use a simpler approach
    
    # Actually, let's try a direct optimization using scipy
    from scipy.optimize import minimize
    
    if time.time() < deadline - 30:
        # Try optimizing a parameterized family
        def neg_score(params):
            h = np.zeros(N)
            # params: log-weights for a subset
            support = params[:N] > -10  # active positions
            active_idx = np.where(support)[0]
            if len(active_idx) == 0:
                return 0.0
            weights = np.exp(params[active_idx])
            weights /= weights.sum()
            h[active_idx] = weights
            return -compute_score(h)
        
        # Try a few initial guesses
        for init_base in [2.0, 3.0, 5.0]:
            if time.time() > deadline:
                break
            x0 = np.zeros(N)
            # Initialize with exponential decay on a Sidon-like set
            for i, idx in enumerate([0, 1, 4, 10, 23]):
                if idx < N:
                    x0[idx] = np.log(init_base ** i)
            
            try:
                res = minimize(neg_score, x0, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-8})
                if res.fun < 0:
                    h_opt = np.zeros(N)
                    support = res.x[:N] > -10
                    active_idx = np.where(support)[0]
                    if len(active_idx) > 0:
                        weights = np.exp(res.x[active_idx])
                        weights /= weights.sum()
                        h_opt[active_idx] = weights
                        sc = compute_score(h_opt)
                        if sc > best_score:
                            best_score = sc
                            best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final: normalize to nonnegative
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
