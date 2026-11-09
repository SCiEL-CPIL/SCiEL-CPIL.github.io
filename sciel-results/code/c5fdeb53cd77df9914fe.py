import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(f):
    """Computes the full linear autoconvolution of f."""
    return np.convolve(f, f, mode='full')

def _compute_ratio(f):
    """Compute max|ac| / (sum f)^2"""
    ac = _autoconv(f)
    s = np.sum(f)
    if abs(s) < 1e-10:
        return np.inf
    return np.max(np.abs(ac)) / (s * s)

def _objective_and_grad(f):
    """
    Objective: log(max |ac|) - 2 log(|sum f|)
    Returns (value, gradient)
    """
    ac = _autoconv(f)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    s = np.sum(f)
    
    if max_val < 1e-30 or abs(s) < 1e-30:
        return 1e10, np.zeros_like(f)
    
    value = np.log(max_val) - 2.0 * np.log(abs(s))
    
    # Find all indices achieving the maximum
    max_indices = np.where(abs_ac >= max_val - 1e-12)[0]
    
    # Subgradient: average over all max indices
    grad = np.zeros_like(f)
    for k in max_indices:
        sgn = 1.0 if ac[k] >= 0 else -1.0
        # d ac[k] / d f[i] = 2 * f[k-i] if 0 <= k-i < N, else 0
        i_vals = np.arange(N)
        j_vals = k - i_vals
        valid = (j_vals >= 0) & (j_vals < N)
        if np.any(valid):
            grad[i_vals[valid]] += 2.0 * sgn * f[j_vals[valid]]
    
    # Average over max indices for a smoother subgradient
    if len(max_indices) > 1:
        grad /= len(max_indices)
    
    # Gradient of log(max|ac|)
    g1 = grad / max_val
    
    # Gradient of -2 log(|sum f|)
    g2 = -2.0 * np.ones(N) / s
    
    return value, g1 + g2

def _local_search_polish(f, max_iters=50, tol=1e-10):
    """
    Local search: try flipping individual signs or small perturbations
    to reduce the ratio.
    """
    best_f = f.copy()
    best_score = _compute_ratio(best_f)
    
    for it in range(max_iters):
        improved = False
        # Try flipping each element
        for i in range(N):
            trial = best_f.copy()
            trial[i] = -trial[i]
            trial_score = _compute_ratio(trial)
            if trial_score < best_score - tol:
                best_f = trial
                best_score = trial_score
                improved = True
                break  # Restart from new position
        
        if not improved:
            # Try small perturbations
            for i in range(N):
                for delta in [-0.1, 0.1]:
                    trial = best_f.copy()
                    trial[i] += delta
                    trial_score = _compute_ratio(trial)
                    if trial_score < best_score - tol:
                        best_f = trial
                        best_score = trial_score
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_f, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s buffer before 150s kill)
    
    def obj_func(f):
        val, _ = _objective_and_grad(f)
        return val
    
    def grad_func(f):
        _, grad = _objective_and_grad(f)
        return grad
    
    best_f = None
    best_score = np.inf
    
    # Initial guesses
    n_restarts = 6
    np.random.seed(42)
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            f0 = np.ones(N)
        elif r == 1:
            f0 = np.random.randn(N)
        elif r == 2:
            f0 = np.random.choice([-1.0, 1.0], size=N)
            if np.sum(f0) < 0:
                f0 = -f0
        elif r == 3:
            # Chirp
            t = np.linspace(0, 10, N)
            f0 = np.real(np.exp(1j * t**2))
        elif r == 4:
            # Block pattern
            f0 = np.tile([1.0, -1.0], N // 2)
        else:
            f0 = np.random.randn(N)
        
        # Optimization
        try:
            res = minimize(
                obj_func, 
                f0, 
                method='L-BFGS-B', 
                jac=grad_func,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.fun < best_score or best_f is None:
                best_f = res.x
                score = _compute_ratio(best_f)
                if score < best_score:
                    best_score = score
                    
        except Exception:
            pass
    
    # If no valid result, return ones
    if best_f is None:
        best_f = np.ones(N)
    
    # Final polish with local search if time permits
    if time.time() < deadline - 20:
        polished_f, polished_score = _local_search_polish(best_f, max_iters=30)
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    return best_f
