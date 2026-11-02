"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _autoconv_max(h):
    """Compute max of discrete autoconvolution of h."""
    n = len(h)
    # Full convolution: result has length 2n-1
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)

def _ratio(h):
    """Compute the objective: ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv_max = _autoconv_max(h)
    if conv_max <= 0:
        return 0.0
    return l2_sq / (l1 * conv_max)

def _negative_ratio(h):
    return -_ratio(h)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Try several initial patterns
    init_patterns = []
    
    # Uniform
    init_patterns.append(np.ones(N))
    
    # Delta at various positions
    for pos in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]:
        d = np.zeros(N)
        d[pos] = 1.0
        init_patterns.append(d)
    
    # Two-point symmetric
    for gap in [1, 2, 3, 5, 10, 20, 25]:
        if gap < N:
            d = np.zeros(N)
            d[0] = 0.5
            d[gap] = 0.5
            init_patterns.append(d.copy())
            # Centered version
            d2 = np.zeros(N)
            mid = N // 2
            if mid - gap//2 >= 0 and mid + gap//2 < N:
                d2[mid - gap//2] = 0.5
                d2[mid + gap//2] = 0.5
                init_patterns.append(d2)
    
    # Block of consecutive ones
    for block_len in [1, 2, 3, 5, 10, 20, 50]:
        if block_len <= N:
            d = np.zeros(N)
            d[:block_len] = 1.0
            init_patterns.append(d)
    
    # Random patterns
    np.random.seed(42)
    for _ in range(20):
        d = np.random.exponential(1.0, size=N)
        init_patterns.append(d)
    
    # Gaussian-like patterns
    for center in [N//4, N//2, 3*N//4]:
        for sigma in [2, 5, 10, 20]:
            x = np.arange(N)
            d = np.exp(-0.5 * ((x - center) / sigma)**2)
            init_patterns.append(d)
    
    # Evaluate all initial patterns
    for pat in init_patterns:
        if time.time() > deadline:
            break
        score = _ratio(pat)
        if score > best_score:
            best_score = score
            best_h = pat.copy()
    
    # Local optimization from best starts
    top_starts = []
    scored = []
    for pat in init_patterns:
        scored.append((_ratio(pat), pat))
    scored.sort(key=lambda x: -x[0])
    top_starts = [s[1] for s in scored[:15]]
    
    for start_pat in top_starts:
        if time.time() > deadline:
            break
        try:
            result = minimize(_negative_ratio, start_pat, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                    'adaptive': True})
            if result.success or result.fun < _negative_ratio(start_pat):
                candidate = np.maximum(0, result.x)  # enforce nonnegativity
                score = _ratio(candidate)
                if score > best_score:
                    best_score = score
                    best_h = candidate.copy()
        except Exception:
            pass
    
    # Second round: more targeted search from best found
    if time.time() < deadline - 10:
        # Try optimizing with penalty for negative values
        for start_pat in top_starts[:5]:
            if time.time() > deadline:
                break
            try:
                def penalized(obj_h):
                    h = np.maximum(0, obj_h)
                    return -_ratio(h) + 1e-6 * np.sum(obj_h[obj_h < 0]**2)
                
                result = minimize(penalized, start_pat, method='Powell',
                                options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-8})
                candidate = np.maximum(0, result.x)
                score = _ratio(candidate)
                if score > best_score:
                    best_score = score
                    best_h = candidate.copy()
            except Exception:
                pass
    
    # Ensure nonnegativity and return
    best_h = np.maximum(0, best_h)
    return list(best_h)
# EVOLVE-BLOCK-END
