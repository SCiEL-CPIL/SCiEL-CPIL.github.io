"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time
import sys

N = 100

# EVOLVE-BLOCK-START
def _overlap_at_shift(h, k):
    """Sum of h[i] * (1 - h[(i+k) mod N]) for i=0..N-1."""
    shifted = np.roll(h, -k)
    return np.sum(h * (1.0 - shifted))

def _max_overlap(h):
    """Maximum over all shifts k of sum h[i] * (1 - h[i+k])."""
    # Compute cross-correlation: c[k] = sum_i h[i] * (1 - h[i+k]) = sum_i h[i] - sum_i h[i]h[i+k]
    # sum_i h[i] = N/2 (fixed). So c[k] = N/2 - autocorr[k].
    # Autocorr can be computed via FFT for all lags at once.
    # c[k] = N/2 - autocorr[k]
    # We want max_k c[k] = N/2 - min_k autocorr[k].
    n = len(h)
    # Zero-pad and use FFT for circular autocorrelation
    h_ext = np.zeros(2 * n)
    h_ext[:n] = h
    f = np.fft.fft(h_ext)
    ac = np.real(np.fft.ifft(f * np.conj(f)))
    # ac[k] = sum_{i=0}^{n-1} h[i] * h[(i+k) mod n] for k=0..n-1
    ac = ac[:n]
    autocorr = ac
    c_vals = n / 2.0 - autocorr
    return np.max(c_vals)

def _max_overlap_grad(h):
    """Compute max overlap and its gradient using a soft-max or exact argmax."""
    n = len(h)
    h_ext = np.zeros(2 * n)
    h_ext[:n] = h
    f = np.fft.fft(h_ext)
    ac = np.real(np.fft.ifft(f * np.conj(f)))[:n]
    c_vals = n / 2.0 - ac
    idx = np.argmax(c_vals)
    # Gradient of c[k] wrt h: c[k] = sum_i h[i](1 - h[i+k])
    # d/dh[j] c[k] = (1 - h[j+k]) - h[j-k]  (cyclic)
    k = idx
    grad = (1.0 - np.roll(h, -k)) - np.roll(h, k)
    return c_vals[idx], grad

def _construct():
    start_time = time.time()
    deadline = 120.0  # 8 seconds margin before 128s hard kill
    max_time = deadline - 8.0  # 112s for computation, 8s margin
    
    # Initial: uniform
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_val = _max_overlap(best_h)
    
    # Try a few random initializations
    best_init_h = h.copy()
    best_init_val = best_val
    rng = np.random.RandomState(42)
    
    # Run optimization from multiple starts
    for trial in range(20):
        if time.time() - start_time > max_time:
            break
        
        if trial == 0:
            h0 = np.full(N, 0.5)
        elif trial == 1:
            h0 = np.concatenate([np.full(N//2, 0.0), np.full(N//2, 1.0)])
        else:
            h0 = rng.random(N)
            h0 *= N / h0.sum()
        
        # Use L-BFGS-B with equality constraint via penalty or use a projected method
        # Better: use scipy optimize with constraint
        
        def objective(x):
            h_local = np.clip(x, 0.0, 1.0)
            val = _max_overlap(h_local)
            return val
        
        # Use SLSQP with bounds and equality constraint
        # Equality: sum(h) = N/2
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                         'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        # Use L-BFGS-B with a penalty for the constraint, then polish
        # Actually, let's use SLSQP
        res = minimize(objective, h0, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
        
        h_opt = np.clip(res.x, 0.0, 1.0)
        # Enforce sum constraint
        h_opt = h_opt + (N/2.0 - h_opt.sum()) / N
        h_opt = np.clip(h_opt, 0.0, 1.0)
        
        # Local polish: project back to satisfy sum exactly
        # Iteratively adjust to get sum exactly N/2 while staying in [0,1]
        for _ in range(1000):
            err = N/2.0 - h_opt.sum()
            if abs(err) < 1e-15:
                break
            mask = (h_opt > 0) & (h_opt < 1)
            if not mask.any():
                break
            h_opt[mask] += err / mask.sum()
            h_opt = np.clip(h_opt, 0.0, 1.0)
        
        val = _max_overlap(h_opt)
        if val < best_init_val:
            best_init_val = val
            best_init_h = h_opt.copy()
    
    best_h = best_init_h.copy()
    best_val = best_init_val
    
    # Final high-precision polish: use the gradient of the max-overlap at the argmax
    # Do a few more optimization steps from best_init_h
    for trial in range(5):
        if time.time() - start_time > max_time:
            break
        
        h0 = best_h + rng.normal(0, 0.01, N)
        h0 = np.clip(h0, 0.0, 1.0)
        h0 = h0 + (N/2.0 - h0.sum()) / N
        h0 = np.clip(h0, 0.0, 1.0)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                         'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        res = minimize(objective, h0, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
        
        h_opt = np.clip(res.x, 0.0, 1.0)
        for _ in range(2000):
            err = N/2.0 - h_opt.sum()
            if abs(err) < 1e-16:
                break
            mask = (h_opt > 0) & (h_opt < 1)
            if not mask.any():
                break
            h_opt[mask] += err / mask.sum()
            h_opt = np.clip(h_opt, 0.0, 1.0)
        
        val = _max_overlap(h_opt)
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
    
    return best_h
# EVOLVE-BLOCK-END

def construct():
    return _construct()
