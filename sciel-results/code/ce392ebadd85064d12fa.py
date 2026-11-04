import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # Safety margin before 150s kill

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for step heights."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        return -1.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-30:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution: (h*h)[k] = sum_i h[i]*h[i+k] for lag k
    # Use FFT for efficiency
    n_fft = 1 << (2*N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)
    # The autoconvolution of h with itself: conv[k] = sum_i h[i]*h[i+k]
    # Sup norm is max over all lags
    sup_conv = np.max(conv)
    if sup_conv < 1e-30:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _neg_ratio(h):
    r = _ratio(h)
    return -r if r > 0 else 1.0

def construct():
    start_time = time.time()
    
    # Try several initial shapes and optimize locally
    best_h = None
    best_ratio = 0.0
    
    # Initial candidates: box, gaussian, triangle, random
    candidates = []
    
    # Box
    box = np.ones(N)
    candidates.append(box.copy())
    
    # Gaussian
    x = np.linspace(-1, 1, N)
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        g = np.exp(-x**2 / (2 * sigma**2))
        candidates.append(g.copy())
    
    # Triangle
    tri = np.maximum(0, 1 - np.abs(x))
    candidates.append(tri.copy())
    
    # Random nonnegative
    np.random.seed(42)
    for _ in range(5):
        r = np.random.rand(N)
        r = r / np.max(r) if np.max(r) > 0 else r
        candidates.append(r.copy())
    
    # Optimize each candidate
    for i, h0 in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Normalize to unit L1 for optimization stability (scale-invariant)
        h0 = h0.copy()
        l1 = np.sum(h0)
        if l1 > 0:
            h0 = h0 / l1
        
        try:
            result = minimize(_neg_ratio, h0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            h_opt = np.maximum(result.x, 0.0)
            r = _ratio(h_opt)
            if r > best_ratio:
                best_ratio = r
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # If we have time, try more random restarts
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 60 and best_h is not None:
        np.random.seed(123)
        for restart in range(20):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            # Random perturbation of best
            h0 = best_h + 0.1 * np.random.randn(N)
            h0 = np.maximum(h0, 0.0)
            l1 = np.sum(h0)
            if l1 > 0:
                h0 = h0 / l1
            
            try:
                result = minimize(_neg_ratio, h0, method='Nelder-Mead',
                                 options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
                h_opt = np.maximum(result.x, 0.0)
                r = _ratio(h_opt)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final check
    if time.time() - start_time > DEADLINE:
        # Return best found so far
        if best_h is not None:
            return [float(x) for x in best_h]
        else:
            return [0.0] * N
    
    if best_h is not None:
        return [float(x) for x in best_h]
    else:
        # Fallback: box
        return [1.0] * N
