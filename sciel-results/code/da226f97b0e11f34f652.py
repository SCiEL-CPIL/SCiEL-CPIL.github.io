import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # Safety margin before 150s kill

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return DEADLINE - _elapsed(start)

def _autoconv(h):
    """Compute discrete autoconvolution of h (length N) -> length 2N-1."""
    return np.convolve(h, h, mode='full')

def _soft_max(conv, beta=50.0):
    """Compute soft-max of conv and its gradient."""
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    sm = max_val + np.log(sum_exp) / beta
    grad_sm = exp_shifted / sum_exp
    return sm, grad_sm

def _ratio(h, beta=50.0):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution).
    h is assumed to be non-negative. The ratio is scale-invariant."""
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv(h_norm)
    sm, _ = _soft_max(conv, beta)
    if sm < 1e-15:
        return 0.0
    return l2_sq / sm

def _local_optimize(h_init, beta=50.0, maxiter=100):
    """Local optimization in h-space using g = sqrt(h) parameterization."""
    g_init = np.sqrt(np.maximum(h_init, 0))
    
    def _obj(g):
        h = g ** 2
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 1e6
        h_norm = h / l1
        l2_sq = np.sum(h_norm ** 2)
        conv = _autoconv(h_norm)
        sm, _ = _soft_max(conv, beta)
        if sm < 1e-15:
            return 1e6
        return -l2_sq / sm
    
    try:
        result = minimize(
            _obj, g_init,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
    except Exception:
        return None
    
    h_opt = result.x ** 2
    return h_opt

def _two_gaussian_profile(sigma1, sigma2, center1, center2, weight=0.5):
    """Profile with two Gaussian pulses at different centers."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h1 = weight * np.exp(-((x - center1) ** 2) / (2 * sigma1 ** 2))
    h2 = (1 - weight) * np.exp(-((x - center2) ** 2) / (2 * sigma2 ** 2))
    return h1 + h2

def _three_gaussian_profile(sigmas, centers, weights):
    """Profile with three Gaussian pulses."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    h = np.zeros(N)
    for s, c, w in zip(sigmas, centers, weights):
        h += w * np.exp(-((x - c) ** 2) / (2 * s ** 2))
    return h

def _sinc_like_profile(f0, width):
    """Sinc-like profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    t = (x - 0) / width
    h = np.sinc(f0 * t) ** 2
    return np.maximum(h, 0)

def _trapezoid_profile(base, top, slope):
    """Trapezoid-like profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    t = np.abs(x)
    h = np.zeros(N)
    mask_outer = t < base
    mask_inner = t < top
    h[mask_outer] = 1.0 - (t[mask_outer] - top) / (base - top) if base > top else 1.0
    h[mask_inner] = 1.0
    return h

def _rectangular_profile(width, center=0.0):
    """Rectangular profile."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    t = np.abs(x - center)
    h = (t < width / 2).astype(float)
    return h

def construct():
    start_time = time.time()
    best_heights = np.ones(N).copy()
    best_score = _ratio(best_heights)
    
    rng = np.random.RandomState(42)
    beta = 50.0
    
    # Strategy 1: Two Gaussian pulses with explicit parameterization
    # This allows the optimizer to exploit orthogonality between pulses
    if _time_left(start_time) > 10:
        for trial in range(20):
            if _time_left(start_time) < 5:
                break
            
            # Random parameters for two Gaussians
            sigma1 = rng.uniform(0.02, 0.12)
            sigma2 = rng.uniform(0.02, 0.12)
            center1 = rng.uniform(-0.2, 0.2)
            center2 = rng.uniform(-0.2, 0.2)
            weight = rng.uniform(0.2, 0.8)
            
            # Ensure centers are sufficiently separated
            if abs(center1 - center2) < 0.05:
                center2 = center1 + rng.choice([-1, 1]) * rng.uniform(0.1, 0.3)
                center2 = np.clip(center2, -0.24, 0.24)
            
            h_init = _two_gaussian_profile(sigma1, sigma2, center1, center2, weight)
            l1 = np.sum(h_init)
            if l1 < 1e-15:
                continue
            h_init = h_init / l1
            
            h_opt = _local_optimize(h_init, beta=beta, maxiter=80)
            if h_opt is not None:
                l1_opt = np.sum(h_opt)
                if l1_opt > 1e-15:
                    h_opt = h_opt / l1_opt
                    score = _ratio(h_opt, beta)
                    if score > best_score:
                        best_score = score
                        best_heights = h_opt.copy()
    
    # Strategy 2: Three Gaussian pulses
    if _time_left(start_time) > 10:
        for trial in range(15):
            if _time_left(start_time) < 5:
                break
            
            sigmas = rng.uniform(0.02, 0.10, size=3)
            centers = rng.uniform(-0.2, 0.2, size=3)
            weights = rng.uniform(0.1, 0.5, size=3)
            weights = weights / np.sum(weights)
            
            h_init = _three_gaussian_profile(sigmas, centers, weights)
            l1 = np.sum(h_init)
            if l1 < 1e-15:
                continue
            h_init = h_init / l1
            
            h_opt = _local_optimize(h_init, beta=beta, maxiter=60)
            if h_opt is not None:
                l1_opt = np.sum(h_opt)
                if l1_opt > 1e-15:
                    h_opt = h_opt / l1_opt
                    score = _ratio(h_opt, beta)
                    if score > best_score:
                        best_score = score
                        best_heights = h_opt.copy()
    
    # Strategy 3: Sinc-like profiles
    if _time_left(start_time) > 8:
        for f0 in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for width in [0.05, 0.1, 0.15, 0.2]:
                if _time_left(start_time) < 3:
                    break
                h_init = _sinc_like_profile(f0, width)
                l1 = np.sum(h_init)
                if l1 < 1e-15:
                    continue
                h_init = h_init / l1
                
                h_opt = _local_optimize(h_init, beta=beta, maxiter=50)
                if h_opt is not None:
                    l1_opt = np.sum(h_opt)
                    if l1_opt > 1e-15:
                        h_opt = h_opt / l1_opt
                        score = _ratio(h_opt, beta)
                        if score > best_score:
                            best_score = score
                            best_heights = h_opt.copy()
    
    # Strategy 4: Rectangular and trapezoid profiles
    if _time_left(start_time) > 8:
        for width in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            for center in [-0.1, 0.0, 0.1]:
                if _time_left(start_time) < 3:
                    break
                h_init = _rectangular_profile(width, center)
                l1 = np.sum(h_init)
                if l1 < 1e-15:
                    continue
                h_init = h_init / l1
                
                h_opt = _local_optimize(h_init, beta=beta, maxiter=40)
                if h_opt is not None:
                    l1_opt = np.sum(h_opt)
                    if l1_opt > 1e-15:
                        h_opt = h_opt / l1_opt
                        score = _ratio(h_opt, beta)
                        if score > best_score:
                            best_score = score
                            best_heights = h_opt.copy()
        
        for base in [0.05, 0.1, 0.15, 0.2]:
            for top in [0.02, 0.05, 0.08]:
                if _time_left(start_time) < 3:
                    break
                h_init = _trapezoid_profile(base, top, 1.0)
                l1 = np.sum(h_init)
                if l1 < 1e-15:
                    continue
                h_init = h_init / l1
                
                h_opt = _local_optimize(h_init, beta=beta, maxiter=40)
                if h_opt is not None:
                    l1_opt = np.sum(h_opt)
                    if l1_opt > 1e-15:
                        h_opt = h_opt / l1_opt
                        score = _ratio(h_opt, beta)
                        if score > best_score:
                            best_score = score
                            best_heights = h_opt.copy()
    
    # Strategy 5: Local optimization from diverse single-Gaussian initializations
    if _time_left(start_time) > 10:
        for sigma in [0.02, 0.03, 0.05, 0.08, 0.1, 0.15]:
            for center in [-0.15, -0.05, 0.0, 0.05, 0.15]:
                if _time_left(start_time) < 3:
                    break
                x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
                h_init = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
                l1 = np.sum(h_init)
                if l1 < 1e-15:
                    continue
                h_init = h_init / l1
                
                h_opt = _local_optimize(h_init, beta=beta, maxiter=50)
                if h_opt is not None:
                    l1_opt = np.sum(h_opt)
                    if l1_opt > 1e-15:
                        h_opt = h_opt / l1_opt
                        score = _ratio(h_opt, beta)
                        if score > best_score:
                            best_score = score
                            best_heights = h_opt.copy()
    
    # Strategy 6: Local optimization from best found so far with perturbations
    if _time_left(start_time) > 10:
        g_best = np.sqrt(best_heights)
        for scale in [0.02, 0.05, 0.1, 0.2]:
            for _ in range(8):
                if _time_left(start_time) < 3:
                    break
                g_pert = g_best + scale * rng.randn(N)
                h_pert = g_pert ** 2
                h_opt = _local_optimize(h_pert, beta=beta, maxiter=40)
                if h_opt is not None:
                    l1_opt = np.sum(h_opt)
                    if l1_opt > 1e-15:
                        h_opt = h_opt / l1_opt
                        score = _ratio(h_opt, beta)
                        if score > best_score:
                            best_score = score
                            best_heights = h_opt.copy()
    
    # Strategy 7: Random initializations
    if _time_left(start_time) > 10:
        for _ in range(20):
            if _time_left(start_time) < 3:
                break
            h_init = np.abs(rng.randn(N))
            l1 = np.sum(h_init)
            if l1 < 1e-15:
                continue
            h_init = h_init / l1
            
            h_opt = _local_optimize(h_init, beta=beta, maxiter=40)
            if h_opt is not None:
                l1_opt = np.sum(h_opt)
                if l1_opt > 1e-15:
                    h_opt = h_opt / l1_opt
                    score = _ratio(h_opt, beta)
                    if score > best_score:
                        best_score = score
                        best_heights = h_opt.copy()
    
    # Final safety: ensure we return valid non-negative heights
    if best_score <= 0:
        best_heights = np.ones(N).copy()
    
    # Ensure all values are non-negative
    best_heights = np.maximum(best_heights, 0)
    
    return best_heights.tolist()
