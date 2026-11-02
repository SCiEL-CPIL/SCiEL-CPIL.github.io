import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50
DEADLINE = 145  # Safety margin from 150s limit
START_TIME = time.time()

def _time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def _autoconvolution_fft(h):
    """Compute autoconvolution using FFT for speed."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    h_padded = np.zeros(size)
    h_padded[:n] = h
    fft_h = np.fft.fft(h_padded)
    conv = np.fft.ifft(fft_h * fft_h).real
    return conv[:2 * n - 1]

def _score(h):
    """Compute the ratio: (sum h^2) / (max autoconvolution) for normalized h."""
    h = np.array(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    total = np.sum(h)
    if total == 0:
        return 0.0
    h_norm = h / total
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconvolution_fft(h_norm)
    max_conv = np.max(conv)
    if max_conv <= 0:
        return 0.0
    return l2_sq / max_conv

def _logsumexp_max(conv, beta):
    """Smooth approximation of max(conv) using log-sum-exp."""
    max_val = np.max(conv)
    if max_val <= 0:
        return 1e-10
    # Soft max: exp(beta * (x - max)) / sum(exp(beta * (x - max))) * max
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -50, 50)  # Prevent overflow
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp == 0:
        return max_val
    # The soft max value
    log_sum_exp = np.log(sum_exp) / beta
    soft_max = max_val + log_sum_exp
    return max(soft_max, 1e-10)

def _smooth_score(h, beta):
    """Score with smooth max approximation."""
    h = np.array(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    total = np.sum(h)
    if total == 0:
        return 0.0
    h_norm = h / total
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconvolution_fft(h_norm)
    max_conv_smooth = _logsumexp_max(conv, beta)
    if max_conv_smooth <= 0:
        return 0.0
    return l2_sq / max_conv_smooth

# EVOLVE-BLOCK-START
def construct():
    def score(h):
        return _score(h)

    # Initial guesses
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Candidate 1: [2/3, 1/3, 0...]
    h1 = np.zeros(N)
    h1[0] = 2/3
    h1[1] = 1/3
    s1 = score(h1)
    if s1 > best_score:
        best_score = s1
        best_h = h1.copy()
        
    # Candidate 2: [2/3, 1/4, 1/4, 0...] (from analysis c=1/4)
    h2 = np.zeros(N)
    h2[0] = 2/3
    h2[1] = 1/3
    h2[2] = 1/4
    # Normalize
    h2 = h2 / np.sum(h2)
    s2 = score(h2)
    if s2 > best_score:
        best_score = s2
        best_h = h2.copy()

    # Penalty method with smooth surrogate for the max autoconvolution
    # We optimize: maximize l2_sq / t subject to t >= max(conv)
    # Using penalty: maximize l2_sq / t - mu * max(0, max(conv) - t)^2
    # With smooth approximation for max(conv)
    
    def penalty_objective(x, mu, beta):
        """Objective for penalty method. x are the heights (first k elements)."""
        h_full = np.zeros(N)
        h_full[:len(x)] = np.abs(x)
        total = np.sum(h_full)
        if total == 0:
            return 1e10
        h_norm = h_full / total
        l2_sq = np.sum(h_norm ** 2)
        conv = _autoconvolution_fft(h_norm)
        max_conv = _logsumexp_max(conv, beta)
        
        # t is the estimated peak; we want t >= max_conv
        # Penalty: mu * (max_conv - t)^2 if max_conv > t
        # We use t = max_conv * (1 + epsilon) as a smooth relaxation
        # Actually, let's just minimize -l2_sq / max_conv_smooth with penalty for constraint violation
        
        # Simpler: just use the smooth score
        return -_smooth_score(h_full, beta)

    # Run optimization with increasing beta (sharpening the max approximation)
    betas = [5, 10, 20, 50, 100, 200]
    current_h = best_h.copy()
    
    for beta in betas:
        if _time_remaining() < 15:
            break
            
        # Optimize using Nelder-Mead on the first 15 dimensions
        def obj(x):
            h_full = np.zeros(N)
            h_full[:len(x)] = np.abs(x)
            if np.sum(h_full) == 0:
                return 1e10
            return -_smooth_score(h_full, beta)
        
        x0 = current_h[:15].copy()
        # Ensure x0 is valid
        if np.sum(np.abs(x0)) == 0:
            x0[0] = 0.5
            x0[1] = 0.5
        
        try:
            res = minimize(obj, x0, method='Nelder-Mead', 
                          options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11,
                                   'adaptive': True})
            if _time_remaining() < 10:
                break
            h_opt = np.zeros(N)
            h_opt[:len(res.x)] = np.abs(res.x)
            s_opt = score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
            current_h = h_opt.copy()
        except Exception:
            pass

    # Final high-precision polish pass with exact max (no smoothing)
    if _time_remaining() > 20:
        def obj_exact(x):
            h_full = np.zeros(N)
            h_full[:len(x)] = np.abs(x)
            if np.sum(h_full) == 0:
                return 1e10
            return -score(h_full)
        
        x0 = best_h[:15].copy()
        if np.sum(np.abs(x0)) == 0:
            x0[0] = 0.5
            x0[1] = 0.5
        
        try:
            res = minimize(obj_exact, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            h_opt = np.zeros(N)
            h_opt[:len(res.x)] = np.abs(res.x)
            s_opt = score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Try a few random restarts with small support if time permits
    for _ in range(3):
        if _time_remaining() < 10:
            break
        # Random sparse vector
        support_size = np.random.randint(2, 6)
        indices = np.random.choice(N, support_size, replace=False)
        indices.sort()
        h_rand = np.zeros(N)
        h_rand[indices] = np.random.rand(support_size)
        h_rand = h_rand / np.sum(h_rand)
        
        x0_rand = h_rand[:15].copy()
        try:
            res = minimize(obj_exact, x0_rand, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            h_opt = np.zeros(N)
            h_opt[:len(res.x)] = np.abs(res.x)
            s_opt = score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Final check and return
    # Ensure non-negative
    best_h = np.abs(best_h)
    if np.sum(best_h) == 0:
        best_h[0] = 1.0
    else:
        best_h = best_h / np.sum(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
