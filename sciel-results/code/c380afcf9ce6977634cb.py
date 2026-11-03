import numpy as np
from scipy.optimize import minimize
import time

N = 50
w = 1.0 / 100.0

def compute_ratio(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-10):
        return 0.0
    h = np.maximum(h, 0.0)
    
    norm1 = np.sum(h) * w
    if norm1 < 1e-15:
        return 0.0
    norm2_sq = np.sum(h**2) * w
    
    a = -0.25 + np.arange(N) * w  # left edges, shape (N,)
    
    # Vectorized autoconvolution evaluation at breakpoints
    # For each pair (i,j), overlap at t is max(0, min(a[i]+w, a[j]-t+w) - max(a[i], a[j]-t))
    # Breakpoints: a[j] - a[i] and a[j] + w - a[i] for all i,j
    
    # Create meshgrid of a
    ai = a[:, None]  # (N,1)
    aj = a[None, :]  # (1,N)
    
    # Breakpoints for each (i,j): aj - ai and aj + w - ai
    bp1 = aj - ai      # (N,N)
    bp2 = aj + w - ai  # (N,N)
    
    all_bp = np.concatenate([bp1.flatten(), bp2.flatten()])
    all_bp = np.unique(all_bp)
    
    # Evaluate (f*f)(t) at each breakpoint
    # (f*f)(t) = sum_{i,j} h[i]*h[j]*max(0, min(ai[i]+w, aj[j]-t+w) - max(ai[i], aj[j]-t))
    
    max_val = 0.0
    for t in all_bp:
        # overlap[i,j] = max(0, min(ai[i]+w, aj[j]-t+w) - max(ai[i], aj[j]-t))
        lo = np.maximum(ai, aj - t)      # (N,N)
        hi = np.minimum(ai + w, aj - t + w)  # (N,N)
        overlap = np.maximum(0, hi - lo)
        val = np.sum(h[:, None] * h[None, :] * overlap)
        if val > max_val:
            max_val = val
    
    if max_val < 1e-15:
        return 0.0
    
    return norm2_sq / (norm1 * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_ratio = compute_ratio(best_h)
    
    bounds = [(0, 10)] * N
    
    def neg_ratio(h):
        return -compute_ratio(h)
    
    np.random.seed(42)
    
    initializations = [
        np.ones(N),
        np.zeros(N) + (np.arange(N) == N//2).astype(float),
        np.zeros(N) + (np.abs(np.arange(N) - N//2) <= 2).astype(float),
        np.zeros(N) + (np.arange(N) % 2 == 0).astype(float),
        np.zeros(N) + (np.arange(N) < N//2).astype(float),
        np.random.exponential(0.5, size=N),
        np.random.exponential(0.5, size=N),
        np.zeros(N) + (np.abs(np.arange(N) - N//2) <= 5).astype(float),
        np.zeros(N) + (np.abs(np.arange(N) - N//2) <= 10).astype(float),
    ]
    
    for i, h_init in enumerate(initializations):
        if time.time() > deadline:
            break
        
        h_init = np.asarray(h_init, dtype=float)
        if np.sum(h_init) < 1e-10:
            h_init = np.ones(N)
        
        try:
            result = minimize(neg_ratio, h_init, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if not np.any(np.isnan(result.x)):
                r = compute_ratio(result.x)
                if r > best_ratio:
                    best_ratio = r
                    best_h = result.x.copy()
        except Exception:
            pass
    
    return [float(x) for x in best_h]
