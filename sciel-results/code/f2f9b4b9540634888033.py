import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _smooth_max_conv(h, beta=100.0):
    """Soft-max surrogate for max of autoconvolution using log-sum-exp."""
    h_conv = np.convolve(h, h, mode='full')
    # log-sum-exp for numerical stability
    m = np.max(h_conv)
    if m <= 0:
        return 0.0
    lse = m + np.log(np.sum(np.exp((h_conv - m) * beta)) / beta)
    # The soft-max approximates max as beta -> inf
    # We want to minimize the peak, so we use this as a smooth surrogate
    return lse / beta  # This is an underestimate of max; use max for scoring

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s kill
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Parametric initializations ---
    candidates = []
    
    # 1. Uniform
    h_unif = np.ones(N)
    candidates.append(h_unif)
    
    # 2. Gaussians
    for c in np.linspace(3, N-3, 15):
        for w in np.linspace(0.5, N/2, 15):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            candidates.append(h_gauss)
    
    # 3. Triangular
    for c in np.linspace(3, N-3, 15):
        for w in np.linspace(1, N, 15):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                candidates.append(h_tri)
    
    # 4. Cosine bumps
    for c in np.linspace(3, N-3, 10):
        for w in np.linspace(2, N/2, 10):
            h_cos = np.maximum(0, np.cos(np.pi * (x_idx - c) / (2*w)))
            if np.sum(h_cos) > 0:
                candidates.append(h_cos)
    
    # 5. Sparse peaks (Sidon-like intuition: minimize autoconvolution overlap)
    for _ in range(200):
        if time.time() > deadline - 30: break
        h_sparse = np.zeros(N)
        num_peaks = np.random.randint(3, 12)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_sparse[indices] = 1.0
        # Add small noise to break symmetry
        h_sparse += np.random.rand(N) * 0.05
        candidates.append(h_sparse)
    
    # 6. Random smooth
    for _ in range(100):
        if time.time() > deadline - 30: break
        h_r = np.random.rand(N)
        kernel = np.ones(np.random.randint(3, 8)) / 5.0
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 0:
            candidates.append(h_r)
    
    # Evaluate all candidates
    best_h = None
    best_score = 0.0
    for h_cand in candidates:
        h_cand = np.asarray(h_cand, dtype=np.float64)
        if np.sum(h_cand) <= 0:
            continue
        s = _score_raw(h_cand)
        if s > best_score:
            best_score = s
            best_h = h_cand.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_score = _score_raw(best_h)
    
    # --- Phase 2: Optimization ---
    # Objective: Maximize L2^2 / (L1 * DELTA * max_conv)
    # Since DELTA is constant, maximize L2^2 / (L1 * max_conv)
    # With L1 fixed to 1, maximize L2^2 / max_conv
    
    def neg_obj_h(h):
        l1 = np.sum(h)
        if l1 <= 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv <= 1e-15:
            return 0.0
        return -(l2_sq / (l1 * max_conv))
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    # Normalize best_h for optimization
    h_start = best_h / np.sum(best_h)
    
    # Try SLSQP from best initialization
    try:
        res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
        if res.success and res.fun < 0:
            h_candidate = np.maximum(0, res.x)
            if np.sum(h_candidate) > 0:
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
    except Exception:
        pass
    
    # Generate diverse initializations for restarts
    inits = []
    
    # 1. Best found so far
    inits.append(best_h / np.sum(best_h))
    
    # 2. Uniform
    inits.append(np.ones(N) / N)
    
    # 3. Top-scoring parametric candidates
    scored_cands = []
    for h_cand in candidates:
        h_cand = np.asarray(h_cand, dtype=np.float64)
        if np.sum(h_cand) <= 0:
            continue
        s = _score_raw(h_cand)
        scored_cands.append((s, h_cand))
    scored_cands.sort(key=lambda x: -x[0])
    for s, h_cand in scored_cands[:20]:
        inits.append(h_cand / np.sum(h_cand))
    
    # 4. Random sparse
    for _ in range(30):
        h_s = np.zeros(N)
        k = np.random.randint(3, 12)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s += np.random.rand(N) * 0.05
        if np.sum(h_s) > 0:
            inits.append(h_s / np.sum(h_s))
    
    # 5. Random smooth
    for _ in range(30):
        h_r = np.random.rand(N)
        kernel = np.ones(np.random.randint(3, 8)) / 5.0
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 0:
            inits.append(h_r / np.sum(h_r))
    
    # Optimize from each init
    for h_init in inits:
        if time.time() > deadline - 15:
            break
        
        if np.sum(h_init) <= 0:
            continue
        h_init = np.maximum(0, h_init)
        h_init = h_init / np.sum(h_init)
        
        try:
            res = minimize(neg_obj_h, h_init, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success and res.fun < 0:
                h_candidate = np.maximum(0, res.x)
                if np.sum(h_candidate) > 0:
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-14})
            if res.success and res.fun < 0:
                h_candidate = np.maximum(0, res.x)
                if np.sum(h_candidate) > 0:
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
        except Exception:
            pass
    
    # Return the optimized shape (normalized to L1=1)
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) > 0:
        best_h = best_h / np.sum(best_h)
    
    return list(best_h)
