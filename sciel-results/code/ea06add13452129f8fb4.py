"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # safety margin before 150s kill


def _value(h):
    """Compute the evaluator's value."""
    s = np.sum(h)
    if s == 0:
        return np.inf
    ac = np.convolve(h, h)
    return abs(2 * N * np.max(ac) / s ** 2)


def _smooth_max(x, beta=100.0):
    """Log-sum-exp smooth approximation of max(x)."""
    m = np.max(x)
    if m < -700:
        return m
    return m + np.log(np.sum(np.exp(x - m))) / beta


def _objective(h, beta=100.0):
    """Smooth surrogate objective for L-BFGS."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = np.convolve(h, h)
    # Normalize: we want to minimize max(ac) / s^2
    # Use smooth max
    sm = _smooth_max(ac, beta)
    val = 2.0 * N * sm / (s * s)
    if not np.isfinite(val):
        return 1e10
    return val


def _make_sine_init(freq_idx, phase=0.0):
    """Create a sine-wave initialization with given frequency index."""
    k = np.arange(N)
    # freq_idx = 1 gives half period (best basic case)
    # Higher frequencies create more oscillations
    h = np.sin(2 * np.pi * freq_idx * k / N + phase)
    # Ensure sum is not zero and normalize
    s = np.sum(h)
    if abs(s) < 1e-10:
        # Add small DC offset to make sum nonzero
        h = h + 0.01
        s = np.sum(h)
    return h / s * np.sqrt(N)  # normalize so sum^2 ~ N


def construct():
    t_start = time.time()
    
    best_h = np.ones(N)
    best_val = _value(best_h)  # Should be 2.0
    
    # Try multiple sine-based initializations
    freq_candidates = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]
    phases = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    
    # First pass: quick evaluation of many initializations
    for freq in freq_candidates:
        for phase in phases:
            if time.time() - t_start > DEADLINE - 60:
                break
            h = _make_sine_init(freq, phase)
            v = _value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Second pass: L-BFGS optimization from best initializations
    # Sort initializations by value and optimize top few
    init_list = []
    for freq in freq_candidates[:6]:
        for phase in phases[:2]:
            h = _make_sine_init(freq, phase)
            v = _value(h)
            init_list.append((v, h))
    
    init_list.sort(key=lambda x: x[0])
    
    # Optimize top 4 initializations with increasing beta
    for i, (v0, h0) in enumerate(init_list[:4]):
        if time.time() - t_start > DEADLINE - 15:
            break
        
        # Use beta that increases for more precision
        beta = 100.0 + i * 200.0
        
        try:
            res = minimize(
                _objective, h0, args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.success or res.fun < best_val:
                h_opt = res.x
                v_opt = _value(h_opt)
                if v_opt < best_val and np.isfinite(v_opt):
                    best_val = v_opt
                    best_h = h_opt.copy()
        except Exception:
            continue
    
    # Third pass: high-precision polish on the best candidate
    if time.time() - t_start < DEADLINE - 10:
        try:
            beta_high = 800.0
            res = minimize(
                _objective, best_h, args=(beta_high,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            if res.fun < best_val:
                h_polish = res.x
                v_polish = _value(h_polish)
                if v_polish < best_val and np.isfinite(v_polish):
                    best_val = v_polish
                    best_h = h_polish.copy()
        except Exception:
            pass
    
    # Final check
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    
    return best_h


# EVOLVE-BLOCK-START
# (The construct function above is the entry point)
# EVOLVE-BLOCK-END
