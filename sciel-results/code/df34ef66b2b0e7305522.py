import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_value(h):
    """Compute the objective value for a given height vector h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h, mode='full')
    m = len(g)
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:-1] = g
    # spacing w
    x = np.linspace(-0.5, 0.5, m + 2)
    w = x[1] - x[0]
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # This is a sum over consecutive pairs
    terms = y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    return l2 / (norm_1 * norm_inf)

def _value_smoothed(h, alpha=1.0):
    """Smoothed version using log-sum-exp for norm_inf to aid optimization."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return -1e10
    g = np.convolve(h, h, mode='full')
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    x = np.linspace(-0.5, 0.5, m + 2)
    w = x[1] - x[0]
    
    terms = y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    
    # Smoothed norm_inf via log-sum-exp
    abs_g = np.abs(g)
    max_g = np.max(abs_g)
    if max_g == 0:
        return -1e10
    # log-sum-exp trick: log(sum(exp(g/alpha))) * alpha
    # We want to approximate max(g) with a smooth function
    scaled = abs_g / alpha
    shifted = scaled - np.max(scaled)
    lse = alpha * (np.max(scaled) + np.log(np.sum(np.exp(shifted))))
    
    if norm_1 == 0:
        return -1e10
    
    return l2 / (norm_1 * lse)

def _neg_value_smoothed(h, alpha=1.0):
    return -_value_smoothed(h, alpha)

def _construct_single(alpha=1.0, init=None, maxiter=2000):
    """Run a single optimization from a given initial point."""
    if init is None:
        init = np.ones(N)
    else:
        init = np.asarray(init, dtype=float)
    
    # Ensure nonnegativity by working in log-space or with bounds
    # We'll use bounds to enforce h >= 0
    bounds = [(0.0, None)] * N
    
    try:
        res = minimize(
            _neg_value_smoothed,
            init,
            args=(alpha,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        best_h = res.x
        # Ensure nonnegative
        best_h = np.maximum(best_h, 0.0)
        return best_h, _compute_value(best_h)
    except Exception:
        return init, _compute_value(init)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Generate diverse initial points
    inits = []
    
    # 1. Uniform (seed)
    inits.append(np.ones(N))
    
    # 2. Gaussian centered
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-x**2 / 0.5) + 0.1)
    
    # 3. Two peaks
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-(x - 0.3)**2 / 0.2) + np.exp(-(x + 0.3)**2 / 0.2) + 0.1)
    
    # 4. Power law (peaked at center)
    x = np.linspace(-1, 1, N)
    inits.append(np.power(1 - np.abs(x), 2) + 0.1)
    
    # 5. Sparse peak
    h_sparse = np.ones(N) * 0.1
    h_sparse[N//2] = 5.0
    inits.append(h_sparse)
    
    # 6. Broad peak
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-x**2 / 2.0) + 0.1)
    
    # 7. Sharp center
    h_sharp = np.ones(N) * 0.05
    h_sharp[N//2-1:N//2+2] = 3.0
    inits.append(h_sharp)
    
    # 8. Random smooth
    rng = np.random.RandomState(42)
    h_rand = rng.rand(N) * 0.5 + 0.5
    # Smooth it a bit
    h_rand = np.convolve(h_rand, np.ones(5)/5, mode='same')
    inits.append(h_rand)
    
    # 9. Cosine taper
    x = np.linspace(-1, 1, N)
    inits.append(np.cos(x * np.pi / 2)**2 + 0.1)
    
    # 10. Inverse cosine
    x = np.linspace(-1, 1, N)
    inits.append(1.0 / (1.0 + 10.0 * x**2) + 0.1)
    
    # Try multiple alpha values for smoothing
    alphas = [0.5, 1.0, 2.0, 5.0]
    
    for alpha in alphas:
        if time.time() > deadline - 5.0:
            break
        for init in inits:
            if time.time() > deadline - 2.0:
                break
            h_candidate, val = _construct_single(alpha=alpha, init=init, maxiter=500)
            if val > best_val:
                best_val = val
                best_h = h_candidate.copy()
    
    # Final polish with tighter alpha and more iterations if time permits
    if time.time() < deadline - 10.0:
        for alpha in [0.1, 0.5, 1.0]:
            if time.time() > deadline - 5.0:
                break
            h_candidate, val = _construct_single(alpha=alpha, init=best_h, maxiter=2000)
            if val > best_val:
                best_val = val
                best_h = h_candidate.copy()
    
    # Ensure the result is valid
    best_h = np.maximum(best_h, 0.0)
    best_h[~np.isfinite(best_h)] = 0.0
    
    return list(best_h)
