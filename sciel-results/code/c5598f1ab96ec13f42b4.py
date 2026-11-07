"""Optimized program for second autocorrelation inequality task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _compute_value(h):
    """Fast computation of the objective value."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)  # 99
    w = 1.0 / (m + 1)  # 1/100
    # y = [0, g[0], ..., g[m-1], 0], length m+2
    # l2 = sum_{i=0}^{m} (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # This equals (w/3) * sum of (y[i]^2 + y[i]*y[i+1] + y[i+1]^2) for i=0..m
    # y[0]=0, y[1..m]=g[0..m-1], y[m+1]=0
    # So terms: i=0: 0 + 0 + g[0]^2 = g[0]^2
    #           i=1..m-1: g[i-1]^2 + g[i-1]*g[i] + g[i]^2
    #           i=m: g[m-1]^2 + g[m-1]*0 + 0 = g[m-1]^2
    # Total = g[0]^2 + g[m-1]^2 + sum_{i=1}^{m-1}(g[i-1]^2 + g[i-1]*g[i] + g[i]^2)
    #        = 2*(g[0]^2 + g[m-1]^2) + sum_{i=1}^{m-2}(g[i]^2 + g[i]*g[i+1] + g[i+1]^2) ... let me just compute directly
    y = np.zeros(m + 2)
    y[1:m+1] = g
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    denom = norm_1 * norm_inf
    if denom < 1e-15:
        return -1.0
    return l2 / denom

def _neg_value(h):
    """Negative for minimization."""
    v = _compute_value(h)
    if v < 0:
        return 1e10
    return -v

def _make_initializations():
    """Generate diverse starting points."""
    inits = []
    # Uniform (seed)
    inits.append(np.ones(N))
    
    # Triangular
    x = np.linspace(-1, 1, N)
    tri = np.maximum(0, 1 - np.abs(x))
    inits.append(tri / tri.max())
    
    # Gaussian
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-4 * x**2)
    inits.append(gauss / gauss.max())
    
    # Raised cosine
    rc = 0.5 * (1 + np.cos(np.pi * x))
    inits.append(rc / rc.max())
    
    # Flat top with smooth edges
    flat = np.ones(N)
    edges = 8
    for i in range(edges):
        flat[i] = 0.5 * (1 + np.cos(np.pi * i / (2*edges)))
        flat[N-1-i] = 0.5 * (1 + np.cos(np.pi * i / (2*edges)))
    inits.append(flat / flat.max())
    
    # Slightly peaked
    x = np.linspace(-1, 1, N)
    peaked = np.exp(-2 * x**2) + 0.3 * np.exp(-8 * x**2)
    inits.append(peaked / peaked.max())
    
    # Step-like: high in middle
    step = np.zeros(N)
    step[10:40] = 1.0
    # Smooth a bit
    for i in range(5):
        step[i+5] = (i+1)/6.0
        step[N-1-i-5] = (i+1)/6.0
    inits.append(step / step.max())
    
    # Random smooth initializations
    rng = np.random.RandomState(42)
    for _ in range(8):
        h = rng.rand(N)
        # Smooth with a small kernel
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        h = np.convolve(h, kernel, mode='same')
        h = np.maximum(h, 0)
        if h.max() > 1e-10:
            h /= h.max()
        inits.append(h)
    
    return inits

def _local_optimize(h0, max_iter=2000, time_limit=8.0):
    """Nelder-Mead local optimization."""
    t0 = time.time()
    result = minimize(
        _neg_value, h0,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True}
    )
    if time.time() - t0 > time_limit:
        # Still use the best found
        pass
    h_opt = np.maximum(result.x, 0.0)
    return h_opt, _compute_value(h_opt)

def construct():
    """Main entry point: return 50 nonnegative step heights."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    inits = _make_initializations()
    
    best_h = inits[0].copy()
    best_val = _compute_value(best_h)
    
    # Local optimize each initialization
    for i, h0 in enumerate(inits):
        if time.time() > deadline - 15:
            break
        h_opt, val = _local_optimize(h0)
        if val > best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Additional random restarts from perturbed best
    rng = np.random.RandomState(123)
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        # Perturb the best solution
        noise = rng.normal(0, 0.1, N)
        h0 = np.maximum(best_h + noise, 0)
        if h0.max() > 1e-10:
            h0 /= h0.max()
        h_opt, val = _local_optimize(h0, max_iter=1500)
        if val > best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Final refinement with tighter tolerance
    if time.time() < deadline - 5:
        h_opt, val = _local_optimize(best_h, max_iter=3000)
        if val > best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Ensure valid output
    best_h = np.maximum(best_h, 0.0)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
