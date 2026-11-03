import numpy as np
import time
import sys
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

def _smooth_score(x, beta=100.0):
    """Smooth surrogate: maximize L2^2 / soft_max(conv(h,h)) with h=exp(x)/sum(exp(x)).
    
    Uses log-sum-exp for numerical stability. The soft-max approximates max(conv)
    smoothly, enabling gradient-based optimization.
    """
    # Normalize: h_i = exp(x_i) / sum(exp(x_j))
    # Work in log-space for stability
    x_max = np.max(x)
    log_h_unnorm = x - x_max
    h_unnorm = np.exp(log_h_unnorm)
    l1_unnorm = np.sum(h_unnorm)
    h = h_unnorm / l1_unnorm  # L1-normalized
    
    l2_sq = np.sum(h**2)
    
    # Compute convolution
    h_conv = np.convolve(h, h, mode='full')
    
    # Soft max via log-sum-exp
    beta_inv = 1.0 / beta
    mc = np.max(h_conv)
    log_sum_exp = mc + np.log(np.sum(np.exp(beta_inv * (h_conv - mc))))
    soft_max = np.exp(log_sum_exp)
    
    if soft_max == 0:
        return 0.0
    
    return l2_sq / soft_max

def _neg_smooth_score(x, beta=100.0):
    return -_smooth_score(x, beta)

def _neg_smooth_score_grad(x, beta=100.0):
    """Numerical gradient of the smooth score (negative)."""
    eps = 1e-7
    grad = np.zeros_like(x)
    f0 = _neg_smooth_score(x, beta)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        grad[i] = (_neg_smooth_score(xp, beta) - f0) / eps
    return grad

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (evaluator kills at 150s)
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # Phase 1: Parametric search for good initializations
    # 1. Uniform
    h_init = np.ones(N) / N
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # 2. Gaussian
    for c in np.linspace(3, N-3, 15):
        for w in np.linspace(1.5, N/2, 15):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_gauss /= np.sum(h_gauss)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # 3. Triangular
    for c in np.linspace(3, N-3, 15):
        for w in np.linspace(2, N, 15):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                h_tri /= np.sum(h_tri)
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()
    
    # 4. Random sparse initialization
    for _ in range(200):
        if time.time() > deadline - 60: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(3, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = np.random.uniform(0.5, 1.5, num_peaks)
        h_rand += np.random.rand(N) * 0.05
        h_rand /= np.sum(h_rand)
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # Phase 2: Optimization using L-BFGS-B on log-scale variables
    # h_i = exp(x_i) / sum(exp(x_j))  (softmax normalization)
    # This removes both the L1 constraint and the non-negativity constraint.
    
    def _h_from_x(x):
        x_max = np.max(x)
        h = np.exp(x - x_max)
        return h / np.sum(h)
    
    def neg_obj_log(x, beta=100.0):
        return -_smooth_score(x, beta)
    
    # Start from best_h
    h_opt_init = best_h.copy()
    h_opt_init[h_opt_init < 1e-10] = 1e-10
    x0 = np.log(h_opt_init)
    
    # First optimization with moderate beta
    beta_start = 50.0
    res = minimize(
        neg_obj_log, x0, method='L-BFGS-B',
        args=(beta_start,),
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8}
    )
    
    h_candidate = _h_from_x(res.x)
    s_candidate = _score_raw(h_candidate)
    if s_candidate > best_score:
        best_score = s_candidate
        best_h = h_candidate
    
    # Phase 3: Multiple restarts with diverse initializations
    inits = []
    
    # 1. Best found so far
    inits.append(best_h.copy())
    
    # 2. Uniform
    inits.append(np.ones(N) / N)
    
    # 3. Gaussians
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(1.5, N/2, 8):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    # 4. Triangles
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(2, N, 8):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 5. Random sparse
    for _ in range(30):
        h_s = np.zeros(N)
        k = np.random.randint(3, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = np.random.uniform(0.5, 1.5, k)
        h_s += np.random.rand(N) * 0.05
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 6. Random smooth
    for _ in range(30):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)

    # Optimize from each init
    for h_init in inits:
        if time.time() > deadline - 20: break
        
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        
        h_safe = h_init.copy()
        h_safe[h_safe < 1e-10] = 1e-10
        x_init = np.log(h_safe)
        
        try:
            res = minimize(
                neg_obj_log, x_init, method='L-BFGS-B',
                args=(beta_start,),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8}
            )
            h_candidate = _h_from_x(res.x)
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate
        except Exception:
            pass
            
    # Phase 4: Final polish with high beta (near-exact max)
    if time.time() < deadline - 5:
        h_start = best_h.copy()
        h_start[h_start < 1e-10] = 1e-10
        x_start = np.log(h_start)
        
        # Use high beta for near-exact max
        beta_high = 500.0
        
        try:
            res = minimize(
                neg_obj_log, x_start, method='L-BFGS-B',
                args=(beta_high,),
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-10}
            )
            h_candidate = _h_from_x(res.x)
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate
        except Exception:
            pass
        
        # One more polish with even higher beta
        if time.time() < deadline - 2:
            h_start = best_h.copy()
            h_start[h_start < 1e-10] = 1e-10
            x_start = np.log(h_start)
            
            beta_very_high = 2000.0
            
            try:
                res = minimize(
                    neg_obj_log, x_start, method='L-BFGS-B',
                    args=(beta_very_high,),
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12}
                )
                h_candidate = _h_from_x(res.x)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
            except Exception:
                pass

    return list(best_h)

# EVOLVE-BLOCK-END
