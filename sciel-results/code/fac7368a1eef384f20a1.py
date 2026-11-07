"""Program for the second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 135.0  # safety margin before 150s kill

def _evaluate(h):
    """Compute the value metric for a given height vector."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return 0.0
    if np.any(h < -1e-12):
        h = np.maximum(h, 0.0)
    g = np.convolve(h, h, mode='full')
    m = len(g)
    w = 1.0 / N
    y = np.zeros(m + 2)
    y[1:-1] = g
    l2 = (w / 3.0) * (np.sum(y[:m+1]**2) + np.sum(y[:m+1] * y[1:m+2]) + np.sum(y[1:m+2]**2))
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-15:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _build_h_from_params(params):
    """
    Build nonnegative heights from parameters.
    params: array of length 2*n_knots
    First n_knots values are knot positions (in [0, N-1])
    Next n_knots values are amplitudes (>= 0)
    """
    n_knots = len(params) // 2
    knots = np.clip(params[:n_knots], 0, N - 1)
    amps = np.maximum(params[n_knots:], 0.0)
    
    h = np.zeros(N)
    x = np.arange(N, dtype=np.float64)
    
    for k in range(n_knots):
        pos = knots[k]
        amp = amps[k]
        if amp <= 0:
            continue
        # Tent with width 3 (extends 1.5 on each side)
        width = 3.0
        left = pos - width / 2
        right = pos + width / 2
        
        mask_left = (x >= left) & (x < pos)
        if np.any(mask_left):
            h[mask_left] += amp * (x[mask_left] - left) / (width / 2)
        
        mask_right = (x >= pos) & (x <= right)
        if np.any(mask_right):
            h[mask_right] += amp * (right - x[mask_right]) / (width / 2)
    
    h = np.maximum(h, 0.0)
    return h

def _optimize_tent(n_knots, init_knots, init_amps, max_iter=300, timeout=10):
    """Optimize tent basis parameters using Nelder-Mead with time limit."""
    x0 = np.concatenate([init_knots, init_amps])
    start = time.time()
    
    def neg_value(params):
        if time.time() - start > timeout:
            return 1e6
        h = _build_h_from_params(params)
        return -_evaluate(h)
    
    try:
        res = minimize(neg_value, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
        best_params = res.x
        best_h = _build_h_from_params(best_params)
        best_val = _evaluate(best_h)
        return best_h, best_val
    except:
        h = _build_h_from_params(x0)
        return h, _evaluate(h)

def _coordinate_ascent_polish(h, max_iter=20, timeout=5):
    """Polish using coordinate ascent."""
    h = np.array(h, dtype=np.float64)
    best_val = _evaluate(h)
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > timeout:
            break
        improved = False
        for i in range(N):
            old = h[i]
            # Try several perturbations
            for delta in [-0.1, -0.05, 0.0, 0.05, 0.1, 0.2]:
                h[i] = max(0.0, old + delta)
                v = _evaluate(h)
                if v > best_val + 1e-10:
                    best_val = v
                    improved = True
            h[i] = old
        if not improved:
            break
    
    return h, best_val

def construct():
    start_time = time.time()
    
    # Seed
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Try various tent configurations
    configs = [
        # (n_knots, init_knots, init_amps)
        (3, np.array([10.0, 25.0, 40.0]), np.array([1.0, 1.0, 1.0])),
        (4, np.array([5.0, 18.0, 32.0, 45.0]), np.array([1.0, 1.0, 1.0, 1.0])),
        (5, np.array([3.0, 13.0, 25.0, 37.0, 47.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0])),
        (6, np.array([2.0, 10.0, 18.0, 32.0, 40.0, 48.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
        (3, np.array([8.0, 25.0, 35.0]), np.array([0.5, 1.0, 0.5])),
        (4, np.array([5.0, 20.0, 30.0, 45.0]), np.array([0.3, 1.0, 1.0, 0.3])),
        (5, np.array([2.0, 12.0, 25.0, 38.0, 48.0]), np.array([0.2, 0.6, 1.0, 0.6, 0.2])),
        (2, np.array([20.0, 30.0]), np.array([1.0, 1.0])),
        (3, np.array([22.0, 25.0, 28.0]), np.array([0.5, 1.0, 0.5])),
        (4, np.array([8.0, 18.0, 32.0, 42.0]), np.array([0.8, 1.0, 1.0, 0.8])),
        (5, np.array([5.0, 15.0, 25.0, 35.0, 45.0]), np.array([0.5, 0.8, 1.0, 0.8, 0.5])),
        (6, np.array([4.0, 12.0, 20.0, 30.0, 38.0, 46.0]), np.array([0.4, 0.7, 1.0, 1.0, 0.7, 0.4])),
    ]
    
    for n_knots, init_knots, init_amps in configs:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        h_tent, val_tent = _optimize_tent(n_knots, init_knots, init_amps, max_iter=200, timeout=8)
        
        if val_tent > best_val:
            best_val = val_tent
            best_h = h_tent.copy()
        
        # Polish if we have time
        if time.time() - start_time < DEADLINE - 20 and val_tent > best_val - 0.02:
            h_polished, val_polished = _coordinate_ascent_polish(h_tent, max_iter=15, timeout=4)
            if val_polished > best_val:
                best_val = val_polished
                best_h = h_polished.copy()
    
    # Final polish on best solution
    if time.time() - start_time < DEADLINE - 5:
        h_final, val_final = _coordinate_ascent_polish(best_h, max_iter=25, timeout=8)
        if val_final > best_val:
            best_val = val_final
            best_h = h_final.copy()
    
    # Ensure validity
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
