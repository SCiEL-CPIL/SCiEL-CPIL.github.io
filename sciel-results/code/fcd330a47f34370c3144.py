"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_ratio(h):
    """Compute the actual ratio for evaluation."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-30:
        return 0.0
    corr = np.correlate(h, h, mode='full')
    peak = np.max(corr)
    if peak < 1e-30:
        return 0.0
    return l2_sq / (l1 * peak)

def _smooth_objective(h, smooth_param=50.0):
    """Smooth negative ratio for optimization."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-30:
        return 1e10
    corr = np.correlate(h, h, mode='full')
    corr = np.clip(corr, 1e-30, None)
    log_corr = np.log(corr)
    max_log = np.max(log_corr)
    smooth_max = max_log + np.log(np.sum(np.exp(log_corr - max_log))) / smooth_param
    peak = np.exp(smooth_max)
    if peak < 1e-30:
        return 1e10
    ratio = l2_sq / (l1 * peak)
    return -ratio

def construct():
    t0 = time.time()
    deadline = t0 + 105.0  # Safety margin
    
    best_val = -np.inf
    best_h = np.ones(N)
    
    # Evaluate seed
    val = _compute_ratio(best_h)
    best_val = val
    
    rng = np.random.default_rng(123)
    
    # Generate diverse initializations
    init_list = []
    x = np.linspace(-1, 1, N)
    
    # Constant
    init_list.append(np.ones(N))
    
    # Gaussians with various widths
    for sigma in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
        init_list.append(np.exp(-x**2 / (2*sigma**2)))
    
    # Bimodal peaks
    for gap in [0.3, 0.5, 0.7, 0.9]:
        for width in [0.15, 0.25, 0.4]:
            h_init = np.exp(-((x-gap)**2) / (2*width**2)) + np.exp(-((x+gap)**2) / (2*width**2))
            init_list.append(h_init)
    
    # Triangular
    init_list.append(1 - np.abs(x))
    
    # Cosine window
    init_list.append(0.5 * (1 + np.cos(np.pi * x)))
    
    # Random smooth
    for _ in range(8):
        h_rand = np.abs(rng.normal(1.0, 0.8, N))
        # Smooth it
        from scipy.ndimage import gaussian_filter1d
        h_rand = gaussian_filter1d(h_rand, sigma=2.0)
        init_list.append(h_rand)
    
    # Sparse-ish
    for _ in range(5):
        h_sparse = np.zeros(N)
        idx = rng.choice(N, size=rng.integers(3, 15), replace=False)
        h_sparse[idx] = rng.uniform(0.5, 1.5, size=len(idx))
        from scipy.ndimage import gaussian_filter1d
        h_sparse = gaussian_filter1d(h_sparse, sigma=1.0)
        h_sparse = np.clip(h_sparse, 0, None)
        if np.sum(h_sparse) > 1e-10:
            init_list.append(h_sparse)
    
    bounds = [(0, None)] * N
    
    for i, h0 in enumerate(init_list):
        if time.time() > deadline:
            break
        
        h0 = np.asarray(h0, dtype=np.float64)
        h0 = np.clip(h0, 0, None)
        s = np.sum(h0)
        if s < 1e-10:
            continue
        h0 = h0 * N / s  # Normalize to sum = N for consistent scale
        
        try:
            res = minimize(
                _smooth_objective, h0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'maxfun': 8000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = np.clip(res.x, 0, None)
            val = _compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final refinement
    if time.time() < deadline - 3:
        try:
            res = minimize(
                _smooth_objective, best_h, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 15000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = np.clip(res.x, 0, None)
            val = _compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    return list(best_h)
# EVOLVE-BLOCK-END
