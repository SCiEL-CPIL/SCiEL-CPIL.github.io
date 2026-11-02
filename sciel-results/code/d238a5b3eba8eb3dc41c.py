import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_max_overlap(h):
    """Compute max cyclic overlap of h with shifted (1-h)."""
    h = np.asarray(h)
    one_minus_h = 1.0 - h
    # For each shift k, compute sum_i h[i] * (1-h[(i+k) % N])
    max_ov = 0.0
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    # Initial guess: uniform 0.5
    h0 = np.full(N, 0.5)
    
    # Try a few different initializations and pick the best
    candidates = [h0.copy()]
    
    # Binary alternating pattern
    h_alt = np.zeros(N)
    h_alt[1::2] = 1.0
    candidates.append(h_alt.copy())
    
    # Random initialization
    rng = np.random.default_rng(42)
    h_rand = rng.random(N)
    # Scale to sum to N/2
    h_rand = h_rand * (N/2.0) / h_rand.sum()
    candidates.append(h_rand.copy())
    
    # Evaluate candidates
    best_h = None
    best_score = float('inf')
    for h in candidates:
        # Ensure sum is exactly N/2
        h = np.clip(h, 0, 1)
        h = h * (N/2.0) / h.sum()
        s = _compute_max_overlap(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Use scipy optimization to refine
    def objective(x):
        x = np.clip(x, 0, 1)
        x = x * (N/2.0) / x.sum()
        return _compute_max_overlap(x)
    
    def objective_grad(x):
        x = np.clip(x, 0, 1)
        x = x * (N/2.0) / x.sum()
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        base = objective(x)
        for i in range(N):
            xp = x.copy()
            xp[i] += eps
            xp = np.clip(xp, 0, 1)
            xp = xp * (N/2.0) / xp.sum()
            xm = x.copy()
            xm[i] -= eps
            xm = np.clip(xm, 0, 1)
            xm = xm * (N/2.0) / xm.sum()
            grad[i] = (objective(xp) - objective(xm)) / (2*eps)
        return grad
    
    # Try L-BFGS-B optimization
    try:
        result = minimize(objective, best_h, method='L-BFGS-B', 
                         bounds=[(0,1)]*N, options={'maxiter': 200, 'ftol': 1e-10})
        if result.success or result.fun < best_score:
            h_opt = np.clip(result.x, 0, 1)
            h_opt = h_opt * (N/2.0) / h_opt.sum()
            s = _compute_max_overlap(h_opt)
            if s < best_score:
                best_score = s
                best_h = h_opt
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_h
    
    # Try a few more random restarts if time permits
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        h_trial = rng.random(N)
        h_trial = np.clip(h_trial, 0, 1)
        h_trial = h_trial * (N/2.0) / h_trial.sum()
        try:
            result = minimize(objective, h_trial, method='L-BFGS-B',
                             bounds=[(0,1)]*N, options={'maxiter': 100, 'ftol': 1e-10})
            if result.fun < best_score:
                h_opt = np.clip(result.x, 0, 1)
                h_opt = h_opt * (N/2.0) / h_opt.sum()
                s = _compute_max_overlap(h_opt)
                if s < best_score:
                    best_score = s
                    best_h = h_opt
        except Exception:
            continue
    
    # Final normalization to ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    best_h = best_h * (N/2.0) / best_h.sum()
    
    return best_h
