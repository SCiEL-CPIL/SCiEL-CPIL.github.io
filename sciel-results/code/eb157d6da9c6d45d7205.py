import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _ratio_from_h(h):
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    f = np.convolve(h, h, mode='full')
    f_inf = np.max(f)
    if f_inf <= 0:
        return 0.0
    return l2_sq / (l1 * f_inf)

def _smooth_ratio(h, temp=0.05):
    l1 = np.sum(h)
    if l1 <= 1e-12:
        return 0.0
    l2_sq = np.sum(h**2)
    f = np.convolve(h, h, mode='full')
    if np.max(f) <= 0:
        return 0.0
    # log-sum-exp for smooth max
    m = np.max(f)
    logsumexp = m + np.log(np.sum(np.exp(f - m)))
    smooth_max = np.exp(logsumexp)
    return l2_sq / (l1 * smooth_max)

def _neg_smooth_ratio(params, temp=0.05):
    h = np.abs(params)
    r = _smooth_ratio(h, temp)
    if r <= 0:
        return 1e10
    return -np.log(r)

def _neg_smooth_ratio_jac(params, temp=0.05):
    h = np.abs(params)
    eps = 1e-7
    grad = np.zeros_like(params)
    base = _neg_smooth_ratio(params, temp)
    for i in range(N):
        p = params.copy()
        p[i] += eps
        grad[i] = (_neg_smooth_ratio(p, temp) - base) / eps
    return grad

def construct():
    start_time = time.time()
    deadline = 145  # 5s safety margin
    
    best_ratio = 0.0
    best_h = np.ones(N)
    
    # Generate diverse initial points
    def _make_initial(seed):
        rng = np.random.RandomState(seed)
        # Mix of different profile types
        t = np.linspace(0, 1, N)
        profile_type = seed % 5
        if profile_type == 0:
            # Power law
            alpha = rng.uniform(0.5, 5.0)
            h = t ** alpha
        elif profile_type == 1:
            # Gaussian bump
            center = rng.uniform(0.2, 0.8)
            width = rng.uniform(0.1, 0.4)
            h = np.exp(-((t - center)**2) / (2 * width**2))
        elif profile_type == 2:
            # Two bumps
            c1 = rng.uniform(0.1, 0.4)
            c2 = rng.uniform(0.6, 0.9)
            w = rng.uniform(0.08, 0.2)
            h = np.exp(-((t - c1)**2) / (2 * w**2)) + np.exp(-((t - c2)**2) / (2 * w**2))
        elif profile_type == 3:
            # Sine-based
            freq = rng.uniform(1, 4)
            phase = rng.uniform(0, np.pi)
            h = np.maximum(0, np.sin(freq * np.pi * t + phase))
        else:
            # Random with smoothing
            h = rng.rand(N)
            h = np.convolve(h, np.ones(3)/3, mode='same')
            h = np.maximum(0, h)
        
        # Normalize to unit L1
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h
    
    # Multi-start optimization
    n_starts = 30
    for i in range(n_starts):
        if time.time() > deadline:
            break
        
        h0 = _make_initial(i)
        
        # Optimize in log-space to maintain non-negativity
        # params = log(h + small_offset)
        offset = 1e-6
        params0 = np.log(h0 + offset)
        
        try:
            result = minimize(
                _neg_smooth_ratio,
                params0,
                jac=_neg_smooth_ratio_jac,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        h_opt = np.exp(result.x) - offset
        h_opt = np.maximum(0, h_opt)
        
        # Also try the original params directly (abs)
        h_opt2 = np.abs(result.x)
        
        for h_candidate in [h_opt, h_opt2]:
            if np.sum(h_candidate) <= 0:
                continue
            r = _ratio_from_h(h_candidate)
            if r > best_ratio:
                best_ratio = r
                best_h = h_candidate.copy()
        
        # Check initial point too
        r0 = _ratio_from_h(h0)
        if r0 > best_ratio:
            best_ratio = r0
            best_h = h0.copy()
    
    # Final refinement: coordinate descent on the best solution
    if best_ratio > 0:
        h_cur = best_h.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try scaling h[i]
                for scale in [0.5, 0.8, 1.2, 1.5, 2.0, 0.2, 3.0]:
                    h_try = h_cur.copy()
                    h_try[i] *= scale
                    r = _ratio_from_h(h_try)
                    if r > best_ratio:
                        best_ratio = r
                        h_cur = h_try
                        improved = True
                        break
    
    best_h = h_cur if 'h_cur' in dir() else best_h
    return list(best_h)
