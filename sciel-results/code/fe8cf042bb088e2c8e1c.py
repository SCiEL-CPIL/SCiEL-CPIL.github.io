"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_bound(h):
    """Compute the upper bound on C3: max |autocorrelation| at non-zero lags / |autocorrelation| at lag 0."""
    R = np.correlate(h, h, mode='full')
    R = R[len(R)//2:]  # non-negative lags
    R0 = R[0]
    if abs(R0) < 1e-15:
        return 1e15
    # Peak sidelobe: max |R[m]| for m != 0
    sidelobes = np.abs(R[1:])
    peak = np.max(sidelobes) if len(sidelobes) > 0 else 0.0
    return peak / abs(R0)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Initial guess: Kaiser window
    beta = 10.0
    h0 = np.kaiser(N, beta)
    h0 = h0 / np.linalg.norm(h0)  # normalize
    
    best_h = h0.copy()
    best_bound = _autoconv_bound(best_h)
    
    # Try a few different beta values
    for beta in [5.0, 8.0, 10.0, 12.0, 15.0]:
        if time.time() > deadline:
            break
        h = np.kaiser(N, beta)
        h = h / np.linalg.norm(h)
        bound = _autoconv_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Local optimization using Nelder-Mead on a subset of parameters
    # Use a parametric representation: h[i] = a * cos(pi*i/(N-1)) + b * sin(2*pi*i/(N-1)) + c
    # Or just do coordinate descent
    
    # Coordinate descent on individual heights
    h = best_h.copy()
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            old_h = h[i]
            # Try a few perturbations
            for delta in [-0.1, -0.05, 0.05, 0.1, -0.2, 0.2]:
                h[i] = old_h + delta
                bound = _autoconv_bound(h)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
                    improved = True
                h[i] = old_h
    
    # Final: try to optimize with scipy if time permits
    if time.time() < deadline - 30:
        def objective(x):
            return _autoconv_bound(x)
        
        # Use a reduced set of parameters: optimize every k-th element
        k = 4
        idx = list(range(0, N, k))
        x0 = h0[idx]
        
        def obj_reduced(x):
            h = np.zeros(N)
            for j, i in enumerate(idx):
                h[i] = x[j]
                # Interpolate to neighbors
                if i + 1 < N:
                    h[i+1] = x[j]
                if i + 2 < N:
                    h[i+2] = x[j]
                if i + 3 < N:
                    h[i+3] = x[j]
            return _autoconv_bound(h)
        
        try:
            res = minimize(obj_reduced, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            if time.time() < deadline:
                h_opt = np.zeros(N)
                for j, i in enumerate(idx):
                    h_opt[i] = res.x[j]
                    if i + 1 < N:
                        h_opt[i+1] = res.x[j]
                    if i + 2 < N:
                        h_opt[i+2] = res.x[j]
                    if i + 3 < N:
                        h_opt[i+3] = res.x[j]
                bound = _autoconv_bound(h_opt)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END
