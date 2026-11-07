"""Second autocorrelation inequality optimizer."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s limit

def compute_value(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)  # spacing: linspace(-0.5, 0.5, m+2) has m+1 intervals over length 1
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def construct():
    start_time = time.time()
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Try various structured initializations
    candidates = [
        np.ones(N),
        np.ones(N) * 0.5,
        np.exp(-np.linspace(0, 1, N)**2 * 2),
        np.exp(-np.linspace(-1, 1, N)**2 * 2),
        np.maximum(0, 1 - np.abs(np.linspace(-1, 1, N))),
        np.ones(N) * (1 + 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, N))),
        np.ones(N) * (1 + 0.3 * np.cos(4 * np.pi * np.linspace(0, 1, N))),
        np.maximum(0, np.cos(np.pi * np.linspace(-1, 1, N))),
        np.maximum(0, 1 - 0.5 * np.abs(np.linspace(-1, 1, N))),
        np.ones(N) * (1 + 0.2 * np.sin(2 * np.pi * np.linspace(0, 1, N))),
    ]
    
    for h0 in candidates:
        h0 = np.clip(h0, 0, None)
        v = compute_value(h0)
        if v > best_val:
            best_val = v
            best_h = h0.copy()
    
    # Use scipy minimize with multiple random starts
    def neg_value(h):
        h = np.clip(h, 0, None)
        return -compute_value(h)
    
    np.random.seed(42)
    n_starts = 20
    for i in range(n_starts):
        if time.time() > DEADLINE - 30:
            break
        # Random perturbation of best or random init
        if i < 5:
            h0 = best_h + np.random.randn(N) * 0.3
        elif i < 10:
            h0 = np.abs(np.random.randn(N)) * 0.5
        elif i < 15:
            h0 = np.ones(N) * (0.5 + 0.5 * np.random.rand(N))
        else:
            h0 = best_h * (1 + 0.2 * np.random.randn(N))
        
        h0 = np.clip(h0, 0, None)
        
        try:
            result = minimize(
                neg_value, h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < neg_value(best_h):
                h_opt = np.clip(result.x, 0, None)
                v = compute_value(h_opt)
                if v > best_val:
                    best_val = v
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < DEADLINE - 10:
        try:
            result = minimize(
                neg_value, best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = np.clip(result.x, 0, None)
            v = compute_value(h_opt)
            if v > best_val:
                best_val = v
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Ensure nonnegative and finite
    best_h = np.clip(best_h, 0, None)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    
    return list(best_h)
