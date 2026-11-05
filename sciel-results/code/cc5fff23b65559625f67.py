"""Autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE = 140.0  # safety margin before 150s kill

def _value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _logsumexp(x, beta=50.0):
    """Soft-max surrogate: approximates max(x) smoothly."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    shifted = beta * (x - m)
    # clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    return m + np.log(np.sum(np.exp(shifted))) / beta

def _objective(x, beta=50.0):
    """Objective with sum(x)=1 constraint handled by parameterization."""
    h = x  # we work in normalized space where sum is enforced
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    # Use soft-max for smoothness during optimization
    soft_max = _logsumexp(conv, beta)
    return 2.0 * N * soft_max / (s * s)

def _objective_exact(x):
    """Exact objective for final polish."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return 2.0 * N * np.max(conv) / (s * s)

def _make_smooth_initial(sigma_frac=0.3):
    """Gaussian bump, normalized to sum=1."""
    x = np.linspace(-1, 1, N)
    sigma = sigma_frac
    h = np.exp(-x**2 / (2 * sigma**2))
    h = h / np.sum(h)
    return h

def _make_fourier_initial(freq=2, phase=0.0, amp=0.3):
    """Low-frequency cosine modulation on a bump."""
    x = np.linspace(-1, 1, N)
    sigma = 0.35
    base = np.exp(-x**2 / (2 * sigma**2))
    mod = 1.0 + amp * np.cos(2 * np.pi * freq * x + phase)
    h = base * mod
    h = np.maximum(h, 0)  # keep positive for stability
    if np.sum(h) < 1e-12:
        h = base
    h = h / np.sum(h)
    return h

def _make_block_initial(n_blocks=4):
    """Block structure with smooth edges."""
    h = np.ones(N)
    block_size = N // n_blocks
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, N)
        # Smooth edge
        edge = min(10, block_size // 4)
        if start > 0 and i > 0:
            h[start:start+edge] = 0.5
        if end < N and i < n_blocks - 1:
            h[end-edge:end] = 0.5
    h = h / np.sum(h)
    return h

def _optimize_from(h0, beta_schedule=None, max_iter=200, tol=1e-8):
    """Run L-BFGS-B with soft-max objective, then polish with exact."""
    if beta_schedule is None:
        beta_schedule = [20.0, 50.0, 100.0, 200.0]
    
    best_h = h0.copy()
    best_val = _value(h0)
    
    for beta in beta_schedule:
        def obj(x):
            return _objective(x, beta)
        
        try:
            res = minimize(obj, h0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
            h_new = res.x
            v = _value(h_new)
            if np.isfinite(v) and v < best_val:
                best_h = h_new.copy()
                best_val = v
            h0 = h_new  # warm start for next beta
        except Exception:
            pass
    
    # Final polish with exact objective
    try:
        res = minimize(_objective_exact, best_h, method='L-BFGS-B',
                      options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10})
        h_new = res.x
        v = _value(h_new)
        if np.isfinite(v) and v < best_val:
            best_h = h_new.copy()
            best_val = v
    except Exception:
        pass
    
    return best_h, best_val

def construct():
    t0 = time.time()
    
    # Generate diverse initial candidates
    candidates = []
    
    # Gaussian bumps with various widths
    for sigma in [0.15, 0.25, 0.3, 0.4, 0.5]:
        candidates.append(_make_smooth_initial(sigma))
    
    # Fourier-modulated bumps
    for freq in [1, 2, 3]:
        for phase in [0.0, np.pi/2, np.pi]:
            for amp in [0.2, 0.4, 0.6]:
                candidates.append(_make_fourier_initial(freq, phase, amp))
    
    # Block structures
    for nb in [2, 3, 4, 5, 6, 8]:
        candidates.append(_make_block_initial(nb))
    
    # Sine wave (positive part)
    x = np.linspace(-1, 1, N)
    for freq in [1, 2, 3]:
        h = np.sin(np.pi * freq * x)
        h = np.maximum(h, 0)
        if np.sum(h) > 1e-12:
            h = h / np.sum(h)
            candidates.append(h)
    
    # Random smooth (Gaussian process-like)
    rng = np.random.default_rng(42)
    for trial in range(5):
        h = rng.standard_normal(N)
        # Smooth with moving average
        kernel = np.ones(20) / 20
        h = np.convolve(h, kernel, mode='same')
        h = np.maximum(h, 0)
        if np.sum(h) > 1e-12:
            h = h / np.sum(h)
            candidates.append(h)
    
    # Evaluate all candidates, keep top few
    scored = []
    for h in candidates:
        v = _value(h)
        if np.isfinite(v):
            scored.append((v, h))
    
    scored.sort(key=lambda x: x[0])
    
    # Take top 8 diverse starts
    top_starts = [h for _, h in scored[:8]]
    
    best_h = np.ones(N) / N
    best_val = _value(best_h)
    
    # Optimize from each top start
    for i, h0 in enumerate(top_starts):
        elapsed = time.time() - t0
        if elapsed > DEADLINE - 15:
            break
        
        # Use fewer iterations for later candidates to save time
        remaining = DEADLINE - 15 - elapsed
        if i < 3:
            max_iter = 150
        elif i < 6:
            max_iter = 80
        else:
            max_iter = 40
        
        h_opt, v_opt = _optimize_from(h0.copy(), max_iter=max_iter)
        
        if v_opt < best_val:
            best_val = v_opt
            best_h = h_opt
    
    # Final local polish from best
    elapsed = time.time() - t0
    if elapsed < DEADLINE - 5:
        try:
            res = minimize(_objective_exact, best_h, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            h_new = res.x
            v_new = _value(h_new)
            if np.isfinite(v_new) and v_new < best_val:
                best_h = h_new
                best_val = v_new
        except Exception:
            pass
    
    return best_h

# Quick sanity check if run directly
if __name__ == '__main__':
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Max conv: {np.max(np.convolve(h, h))}")
