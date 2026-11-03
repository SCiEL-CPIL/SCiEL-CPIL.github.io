import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

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

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Gaussians - various centers and widths
    for c in np.linspace(5, N-5, 8):
        for w in np.linspace(2, N/2, 8):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 0:
                inits.append(h_g / np.sum(h_g))
    
    # 3. Triangular
    for c in np.linspace(5, N-5, 8):
        for w in np.linspace(2, N, 8):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                inits.append(h_t / np.sum(h_t))
    
    # 4. Sparse peaks (Sidon-like intuition)
    for _ in range(30):
        h_s = np.zeros(N)
        k = np.random.randint(4, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s += np.random.rand(N) * 0.05
        inits.append(h_s / np.sum(h_s))
    
    # 5. Random smooth
    for _ in range(30):
        h_r = np.random.rand(N)
        kernel = np.ones(3)/3
        h_r = np.convolve(h_r, kernel, mode='same')
        inits.append(h_r / np.sum(h_r))
    
    # 6. Edge-decay patterns
    for alpha in np.linspace(0.5, 3.0, 10):
        h_e = np.exp(-alpha * np.abs(x_idx - N/2) / N)
        inits.append(h_e / np.sum(h_e))
    
    # Objective function: maximize L2^2 / max(conv) subject to L1=1
    def neg_obj_h(h):
        l1 = np.sum(h)
        if l1 == 0:
            return 0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0:
            return 0
        return -(l2_sq / max_conv)
    
    # Smooth surrogate using log-sum-exp for max(conv)
    def neg_obj_smooth(h, beta=50.0):
        l1 = np.sum(h)
        if l1 == 0:
            return 0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        # Soft max using log-sum-exp
        m = np.max(h_conv)
        log_sum_exp = m + np.log(np.sum(np.exp(h_conv - m)))
        if log_sum_exp == 0:
            return 0
        return -(l2_sq / log_sum_exp)
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h}
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    # Optimize from each initialization
    for h_init in inits:
        if time.time() > deadline:
            break
        
        if np.sum(h_init) == 0:
            continue
        h_init = h_init / np.sum(h_init)
        
        # Try smooth surrogate first
        try:
            res = minimize(
                lambda h: neg_obj_smooth(h, beta=50.0),
                h_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
        
        # Try exact objective
        try:
            res = minimize(
                neg_obj_h,
                h_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(
                neg_obj_h,
                h_start,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
    
    # Ensure nonnegativity and normalization
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) == 0:
        best_h = np.ones(N) / N
    else:
        best_h /= np.sum(best_h)
    
    return list(best_h)
