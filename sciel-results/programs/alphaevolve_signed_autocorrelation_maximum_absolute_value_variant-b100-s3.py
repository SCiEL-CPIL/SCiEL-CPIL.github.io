"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_objective(h):
    """Compute max_k |(h*h)[k]| / (sum(h))^2"""
    S = np.sum(h)
    if abs(S) < 1e-12:
        return np.inf
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_abs = np.max(np.abs(conv))
    return max_abs / (S * S)

def _autoconv_max_abs(h):
    """Compute max_k |(h*h)[k]|"""
    conv = np.convolve(h, h, mode='full')
    return np.max(np.abs(conv))

def construct():
    """
    Returns the sequence of signed step heights that minimizes:
    max_k |(h*h)[k]| / (sum(h))^2
    
    Strategy: Use scipy SLSQP to maximize (sum h)^2 subject to |conv_k| <= 1
    for all k. This is equivalent to minimizing max|conv|/S^2. We also try
    local search on the result.
    """
    deadline = time.time() + 135  # Safety margin before 150s limit
    
    # Start with constant vector
    best_h = np.ones(N)
    best_obj = _compute_objective(best_h)
    
    # Try scipy optimization: maximize S^2 subject to |conv_k| <= 1
    # Reformulate: minimize -S^2 subject to conv_k^2 - 1 <= 0 for all k
    # But this has 2N-1 constraints, which is a lot. Use penalty method instead.
    
    def _penalty_objective(h, mu):
        S = np.sum(h)
        conv = np.convolve(h, h, mode='full')
        # Penalty for violating |conv_k| <= 1
        viol = np.maximum(0, np.abs(conv) - 1.0)
        penalty = mu * np.sum(viol ** 2)
        # We want to maximize S^2, so minimize -S^2
        return -S * S + penalty
    
    rng = np.random.RandomState(42)
    
    # Multiple restarts with different initializations
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        if trial == 0:
            h0 = np.ones(N)
        elif trial == 1:
            h0 = rng.choice([-1.0, 1.0], size=N).astype(float)
        else:
            # Mix: mostly positive with some random signs
            h0 = np.ones(N)
            n_flips = rng.randint(1, N // 2)
            flip_idx = rng.choice(N, size=n_flips, replace=False)
            h0[flip_idx] = -1.0
        
        # Use penalty method with increasing mu
        best_local_h = h0.copy()
        best_local_obj = _compute_objective(h0)
        
        for mu in [1.0, 10.0, 100.0, 1000.0]:
            if time.time() > deadline - 10:
                break
            
            def obj_func(x):
                return _penalty_objective(x, mu)
            
            # Bounds: keep values in [-2, 2] to avoid blowup
            bounds = [(-2.0, 2.0)] * N
            
            try:
                result = minimize(obj_func, best_local_h, method='L-BFGS-B',
                                  bounds=bounds,
                                  options={'maxiter': 200, 'ftol': 1e-8})
                if result.success or result.fun < _compute_objective(best_local_h):
                    new_h = result.x
                    new_obj = _compute_objective(new_h)
                    if new_obj < best_local_obj:
                        best_local_obj = new_obj
                        best_local_h = new_h.copy()
            except Exception:
                pass
        
        if best_local_obj < best_obj:
            best_obj = best_local_obj
            best_h = best_local_h.copy()
        
        # Local search: coordinate descent with sign flips
        h = best_local_h.copy()
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            obj = _compute_objective(h)
            
            # Try flipping individual signs
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                h[i] = -h[i]
                new_obj = _compute_objective(h)
                if new_obj < obj:
                    obj = new_obj
                    improved = True
                else:
                    h[i] = -h[i]  # revert
            
            if obj < best_obj:
                best_obj = obj
                best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
