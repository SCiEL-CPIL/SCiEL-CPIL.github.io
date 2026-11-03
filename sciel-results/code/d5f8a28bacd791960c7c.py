import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the score: L2^2 / (L1 * Sup(autoconv))
    For step heights h on intervals of width DELTA:
    L1 = DELTA * sum(h)
    L2^2 = DELTA * sum(h^2)
    autoconv_max = DELTA^2 * max(conv(h, h))
    
    Ratio = (DELTA * sum(h^2)) / (DELTA * sum(h) * DELTA^2 * max_conv)
          = sum(h^2) / (DELTA^2 * sum(h) * max_conv)
    
    Since DELTA is constant, maximizing the raw score is equivalent to maximizing
    J(h) = sum(h^2) / (sum(h) * max_conv)
    
    If we normalize sum(h) = 1, we maximize sum(h^2) / max_conv.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    # The raw score includes the 1/DELTA^2 factor
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # We want to maximize J(h) = sum(h^2) / (sum(h) * max_conv).
    # Let's fix sum(h) = 1. Then we want to maximize sum(h^2) / max_conv.
    # This is equivalent to minimizing max_conv / sum(h^2).
    # max_conv = max_k (h*h)[k] = max_k sum_i h_i h_{k-i} (with appropriate indices)
    # Note that (h*h)[k] is a quadratic form in h.
    # Specifically, let C be the convolution matrix such that conv = C h.
    # Then max_conv = max_k (C h)_k.
    # We want to minimize max_k (C h)_k subject to sum(h)=1, h>=0.
    # This is a convex optimization problem because max of linear functions is convex,
    # but here the functions are quadratic. Wait.
    # (C h)_k = sum_i h_i (C_{ki}).
    # C_{ki} = 1 if i and k-i are valid indices?
    # Actually, conv(h, h)[k] = sum_{i} h_i h_{k-i}.
    # This is quadratic in h. Minimizing the maximum of quadratic functions is not generally convex.
    # However, if we fix the direction of h, or use a different approach...
    
    # Alternative:
    # Maximize sum(h^2) subject to max_conv <= 1, sum(h) = 1, h >= 0.
    # This is a non-convex problem (maximizing convex function).
    
    # Let's try a different structural change:
    # The problem is related to Sidon sets or Golomb rulers.
    # If h is a delta function (one non-zero element), sum(h^2)=1, max_conv=1, score=1.
    # If h is uniform, sum(h^2)=1/N, max_conv=1/N (approx), score ~ 1/N.
    # We want to maximize sum(h^2) while keeping max_conv small.
    # This suggests spreading mass out to reduce peak convolution, but concentrating mass to increase sum(h^2).
    # There is a trade-off.
    
    # Let's use a heuristic search combined with local optimization.
    # We will search for sparse supports and optimize the weights.
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Strategy:
    # 1. Try various sparse supports.
    # 2. For each support, optimize the weights using SLSQP or similar.
    # 3. Also try random dense supports.
    
    # Optimization function for fixed support?
    # No, let's just use SLSQP on the full vector with bounds.
    
    def neg_obj(h):
        l1 = np.sum(h)
        if l1 < 1e-12: return 0
        l2_sq = np.sum(h**2)
        hc = np.convolve(h, h, mode='full')
        mc = np.max(hc)
        if mc < 1e-12: return 0
        return -(l2_sq / mc)
        
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sparse peaks (Sidon-like)
    # A Sidon set has distinct pair sums. This minimizes collisions in convolution.
    # Let's generate some random sparse vectors.
    for _ in range(200):
        k = np.random.randint(3, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 3. Random
    for _ in range(200):
        h_r = np.random.rand(N)
        h_r /= np.sum(h_r)
        inits.append(h_r)
        
    # 4. Structured: powers of 2 indices?
    # Indices: 1, 2, 4, 8, 16, 32
    for _ in range(50):
        h_s = np.zeros(N)
        k = np.random.randint(3, 10)
        # Generate random distinct indices
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # Optimize from each init
    for h_init in inits:
        if time.time() > deadline: break
        try:
            res = minimize(neg_obj, h_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
            if res.success:
                h_candidate = res.x
                # Ensure non-negativity
                h_candidate = np.clip(h_candidate, 0, None)
                # Renormalize
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
            
    # Refine best
    if time.time() < deadline - 5:
        try:
            res = minimize(neg_obj, best_h, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                h_candidate = res.x
                h_candidate = np.clip(h_candidate, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
            
    # Try a few more random restarts if time permits
    while time.time() < deadline - 10:
        h_r = np.random.rand(N)
        h_r /= np.sum(h_r)
        try:
            res = minimize(neg_obj, h_r, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-9})
            if res.success:
                h_candidate = res.x
                h_candidate = np.clip(h_candidate, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
            
    return list(best_h)
