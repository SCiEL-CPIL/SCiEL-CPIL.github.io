"""Optimized program for the second autocorrelation inequality task."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 50

# EVOLVE-BLOCK-START

def _autoconvolution(h):
    """Compute the autoconvolution of h (length N) -> length 2N-1."""
    return np.convolve(h, h, mode='full')

def _score_heights(h):
    """
    Compute the ratio: ||h||_2^2 / (||h||_1 * sup|autoconv(h)|)
    Higher is better.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        # Penalty for negative values
        return -1e10
    h = np.maximum(h, 0.0)
    
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    
    if l1 < 1e-12:
        return -1e10
    
    ac = _autoconvolution(h)
    sup_ac = np.max(np.abs(ac))
    
    if sup_ac < 1e-12:
        return -1e10
    
    ratio = l2_sq / (l1 * sup_ac)
    return ratio

def _negative_score(params):
    """Negative score for minimization."""
    h = np.exp(params)  # Ensure non-negativity
    return -_score_heights(h)

def _polish(h_init):
    """Local refinement using L-BFGS-B on log-heights."""
    h_init = np.maximum(h_init, 1e-10)
    x0 = np.log(h_init)
    
    def obj(x):
        h = np.exp(x)
        return -_score_heights(h)
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
        if res.success:
            h_opt = np.exp(res.x)
            return h_opt, -res.fun
    except Exception:
        pass
    return h_init, _score_heights(h_init)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_h = np.ones(N)
    best_score = _score_heights(best_h)
    
    # Strategy: Differential Evolution on log-heights
    # Bounds for log(h): h in [1e-4, 100] -> log in [-9, 4.6]
    lower_bound = np.full(N, -9.0)
    upper_bound = np.full(N, 4.6)
    
    # Use a subset of N to keep DE fast? No, N=50 is small enough.
    # But DE with 50 vars is slow. Let's try reducing effective dimension or using fewer generations.
    # Alternatively, we can optimize a smaller set of "active" indices?
    # The prompt suggests sparse polynomials/Sidon sets.
    # Let's try a hybrid: Random restarts of DE on subsets or full with low pop.
    
    # Given time constraints, let's run DE with modest settings.
    # Pop size ~ 15 * N = 750. 50 iterations = 37500 evals.
    # Each eval is O(N^2) ~ 2500 ops. Total ~ 10^8 ops. Should be < 10s.
    
    try:
        result = differential_evolution(
            _negative_score,
            bounds=list(zip(lower_bound, upper_bound)),
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            polish=True,
            init='sobol',
            updating='immediate'
        )
        
        if time.time() > deadline - 10:
            # Check if result is available
            if result.success:
                h_de = np.exp(result.x)
                s_de = -result.fun
                if s_de > best_score:
                    best_score = s_de
                    best_h = h_de
        else:
            if result.success:
                h_de = np.exp(result.x)
                s_de = -result.fun
                if s_de > best_score:
                    best_score = s_de
                    best_h = h_de
    except Exception:
        pass
    
    # Polish the best result found
    if time.time() < deadline - 5:
        h_polished, s_polished = _polish(best_h)
        if s_polished > best_score:
            best_score = s_polished
            best_h = h_polished
    
    # Try a few random restarts if time permits
    while time.time() < deadline - 10:
        # Random sparse initialization
        active_count = np.random.randint(5, 15)
        indices = np.random.choice(N, size=active_count, replace=False)
        h_rand = np.zeros(N)
        h_rand[indices] = np.random.uniform(1.0, 10.0, size=active_count)
        
        # Quick local polish
        h_p, s_p = _polish(h_rand)
        if s_p > best_score:
            best_score = s_p
            best_h = h_p
    
    return list(best_h)

# EVOLVE-BLOCK-END
