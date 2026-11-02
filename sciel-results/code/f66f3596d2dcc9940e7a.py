import numpy as np
from scipy.optimize import minimize
import time

N = 50
_TIME_LIMIT = 135.0
_start_time = time.time()

def _score(h):
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _neg_score(h):
    return -_score(h)

def _optimize_from(h0):
    h0 = np.array(h0, dtype=float)
    h0 = np.clip(h0, 0, None)
    if np.sum(h0) < 1e-12:
        return h0, 0.0
    
    # Normalize to sum=1 to reduce scale invariance issues
    h0 = h0 / np.sum(h0)
    
    # Use L-BFGS-B
    # Bounds: 0 to 1 (since sum=1, max element is 1)
    bounds = [(0, 1)] * N
    
    try:
        res = minimize(_neg_score, h0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
        h_opt = res.x
        h_opt = np.clip(h_opt, 0, None)
        # Renormalize
        if np.sum(h_opt) > 1e-12:
            h_opt = h_opt / np.sum(h_opt)
        s_opt = _score(h_opt)
        return h_opt, s_opt
    except Exception:
        return h0, _score(h0)

def construct():
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Diverse initializations
    inits = []
    
    # 1. Two-point (known high performer)
    h1 = np.zeros(N); h1[0] = 2/3; h1[1] = 1/3
    inits.append(h1)
    
    # 2. Uniform
    h2 = np.ones(N) / N
    inits.append(h2)
    
    # 3. Triangular
    h3 = np.zeros(N)
    mid = N // 2
    for i in range(N):
        h3[i] = max(0, 1 - abs(i - mid) / mid)
    inits.append(h3)
    
    # 4. Gaussian
    h4 = np.zeros(N)
    mid = (N - 1) / 2.0
    for i in range(N):
        h4[i] = np.exp(-((i - mid) ** 2) / (2 * (N/4) ** 2))
    inits.append(h4)
    
    # 5. Random
    rng = np.random.default_rng(42)
    h5 = rng.random(N)
    inits.append(h5)
    
    # 6. Two-bump
    h6 = np.zeros(N)
    h6[10] = 1.0
    h6[30] = 1.0
    inits.append(h6)

    for h0 in inits:
        if time.time() - _start_time > _TIME_LIMIT:
            break
        
        h_opt, s_opt = _optimize_from(h0)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt

    # Final Polish: Coordinate Descent
    # We can do a few steps of coordinate descent on the best candidate
    h_best = best_h.copy()
    for _ in range(100):
        if time.time() - _start_time > _TIME_LIMIT:
            break
        improved = False
        for i in range(N):
            # Try to adjust h[i]
            # Since scale invariant, fix sum=1
            # If we change h[i], we must renormalize
            # This is complex. Instead, use local random search with very small steps
            pass
        
        # Simple random polish
        rng = np.random.default_rng(int(time.time() * 1000) % 10000)
        h_pert = h_best + rng.normal(0, 0.01, N)
        h_pert = np.clip(h_pert, 0, None)
        if np.sum(h_pert) > 1e-12:
            h_pert = h_pert / np.sum(h_pert)
            s_pert = _score(h_pert)
            if s_pert > best_score:
                best_score = s_pert
                best_h = h_pert
                improved = True
        if not improved:
            break

    return [float(x) for x in best_h]
