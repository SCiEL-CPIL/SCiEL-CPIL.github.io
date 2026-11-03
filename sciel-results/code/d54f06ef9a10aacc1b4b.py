import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]
CONV_LEN = 2 * N - 1  # Length of full convolution

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

def _compute_conv(h):
    return np.convolve(h, h, mode='full')

def _optimize_for_peak(k_star, h_init, max_iter=200):
    """
    Optimize h to maximize S/L1 subject to c_{k_star} being the max autoconvolution value.
    We normalize c_{k_star} = 1 to fix scale? No, ratio is scale invariant.
    We can fix L1 = 1 and maximize S / max(c).
    Or fix c_{k_star} = 1 and maximize S / L1.
    Let's fix c_{k_star} = 1.
    Objective: maximize S / L1.
    Constraints:
    1. c_{k_star} = 1
    2. c_k <= 1 for all k != k_star
    3. h >= 0
    """
    def objective(x):
        h = np.exp(x)
        # Normalize such that c_{k_star} = 1?
        # It's easier to just maximize S/L1 and let the constraints handle the rest.
        # But if we don't fix scale, it's unbounded?
        # S/L1 is scale invariant.
        # However, c_k <= 1 is not scale invariant.
        # So we MUST fix a scale.
        # Let's fix L1 = 1.
        # Then maximize S / max(c).
        # But max(c) is non-smooth.
        
        # Alternative: Fix c_{k_star} = 1.
        # Then maximize S / L1.
        # Constraints: c_k <= 1 for all k.
        
        conv = _compute_conv(h)
        # Enforce c_{k_star} = 1 via constraint
        # We want to maximize S / L1
        l1 = np.sum(h)
        if l1 == 0: return 0.0
        s = np.sum(h**2)
        return -(s / l1)
        
    # This formulation is tricky because we need to scale h to satisfy c_{k_star}=1.
    # If we use log-variables, h = exp(x).
    # We can add a penalty or use a specific constraint.
    
    # Let's try a different approach:
    # Maximize S / (L1 * M) directly is hard.
    # Let's stick to the previous successful structure but refine it.
    # The previous structure used SLSQP on log-h.
    
    # Let's try to optimize h directly with constraints:
    # 1. h >= 0
    # 2. c_{k_star} - c_k >= 0 for all k (ensuring k_star is the max)
    # 3. Fix scale: sum(h) = 1
    # Objective: Maximize S / M. Since M = c_{k_star} (due to constraint 2),
    # we maximize S / c_{k_star}.
    
    def obj(x):
        h = np.asarray(x)
        if np.any(h < 0): return 1e6
        conv = _compute_conv(h)
        m = conv[k_star]
        if m < 1e-12: return 1e6
        s = np.sum(h**2)
        return -(s / m)
        
    # Constraints
    cons = []
    # sum(h) = 1
    cons.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    
    # c_{k_star} >= c_k for all k
    for k in range(CONV_LEN):
        if k == k_star: continue
        def make_ineq(kk):
            def fun(x):
                h = np.asarray(x)
                if np.any(h < 0): return -1e6
                conv = _compute_conv(h)
                return conv[k_star] - conv[kk]
            return fun
        cons.append({'type': 'ineq', 'fun': make_ineq(k)})
        
    # h >= 0 handled by bounds
    bounds = [(0, None) for _ in range(N)]
    
    # Initial guess
    # Symmetric around k_star/2
    h0 = h_init.copy()
    if np.sum(h0) != 1:
        h0 /= np.sum(h0)
        
    try:
        res = minimize(obj, h0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': max_iter, 'ftol': 1e-12})
        if res.success and res.fun < -1e5:
            h_opt = res.x
            if np.any(h_opt < 0):
                h_opt = np.clip(h_opt, 0, None)
                h_opt /= np.sum(h_opt)
            return h_opt
    except Exception:
        pass
    return None

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Generate a few diverse initializations
    inits = []
    inits.append(np.ones(N) / N)
    
    # Gaussians
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(2, N/3, 5):
            h_g = np.exp(-0.5 * ((np.arange(N) - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    # Bimodal
    for c1 in np.linspace(2, N/4, 3):
        for c2 in np.linspace(3*N/4, N-2, 3):
            h_b = np.exp(-0.5 * ((np.arange(N) - c1) / 2)**2) + np.exp(-0.5 * ((np.arange(N) - c2) / 2)**2)
            h_b /= np.sum(h_b)
            inits.append(h_b)
            
    # Random sparse
    for _ in range(10):
        h_r = np.zeros(N)
        idx = np.random.choice(N, 5, replace=False)
        h_r[idx] = 1.0
        h_r /= np.sum(h_r)
        inits.append(h_r)
        
    # Evaluate inits
    for h_init in inits:
        s = _score_raw(h_init)
        if s > best_score:
            best_score = s
            best_h = h_init.copy()
            
    # Structural change: Iterate over peak indices
    # We only need to check k_star in [0, N-1] due to symmetry
    for k_star in range(N):
        if time.time() > deadline - 10:
            break
            
        # Try to optimize for this peak index using the best_h as init
        # Also try a specific init centered at k_star
        h_init_peak = np.zeros(N)
        center = k_star / 2.0
        # If k_star is small, center is small.
        # We want support such that autoconvolution peaks at k_star.
        # If we put mass at i and k_star-i, it contributes to c_{k_star}.
        # Simple init: mass at floor(k_star/2) and ceil(k_star/2)
        i1 = int(np.floor(k_star / 2.0))
        i2 = int(np.ceil(k_star / 2.0))
        if i1 < N: h_init_peak[i1] = 0.5
        if i2 < N and i2 != i1: h_init_peak[i2] = 0.5
        if np.sum(h_init_peak) > 0:
            h_init_peak /= np.sum(h_init_peak)
        else:
            h_init_peak = best_h.copy()
            
        h_opt = _optimize_for_peak(k_star, h_init_peak, max_iter=100)
        if h_opt is not None:
            s = _score_raw(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
                
        # Also try with best_h as init, in case the peak moved
        if time.time() < deadline - 5:
            h_opt2 = _optimize_for_peak(k_star, best_h, max_iter=100)
            if h_opt2 is not None:
                s = _score_raw(h_opt2)
                if s > best_score:
                    best_score = s
                    best_h = h_opt2.copy()
                    
    return list(best_h)

# EVOLVE-BLOCK-END
