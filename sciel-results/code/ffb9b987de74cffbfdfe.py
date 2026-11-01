"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400
DEADLINE = 140.0  # Safety margin before 150s limit

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _raw_value(h):
    """
    Compute the actual raw value (upper bound on C3) for a given height vector h.
    Metric: max_k |conv(h, h)[k]| / (sum_i |h_i|)^2
    """
    if len(h) == 0:
        return np.inf
    conv = np.convolve(h, h, mode='full')
    max_abs_conv = np.max(np.abs(conv))
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h == 0:
        return np.inf
    return max_abs_conv / (sum_abs_h ** 2)


def _smooth_objective(x, T=5.0):
    """
    Smooth approximation of the objective for gradient-based optimization.
    x is the parameter vector, h = tanh(x) in (-1, 1).
    """
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    
    # Smooth absolute value
    eps = 1e-10
    abs_conv = np.sqrt(conv**2 + eps)
    
    # Smooth max using log-sum-exp
    max_abs = np.max(abs_conv)
    log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
    smooth_max = T * log_sum_exp
    
    # Smooth sum of absolute values
    sum_abs_h = np.sum(np.sqrt(h**2 + eps))
    
    value = smooth_max / (sum_abs_h ** 2)
    return value


def _optimize_from(x0, max_iter=500):
    """Run L-BFGS-B from a given initial condition and return the optimized h."""
    res = minimize(
        _smooth_objective, 
        x0, 
        method='L-BFGS-B', 
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6}
    )
    return np.tanh(res.x)


def construct():
    start_time = time.time()
    
    # Track best solution found so far
    best_h = np.ones(N)
    best_val = _raw_value(best_h)
    
    # 1. Primary optimization from seed (all ones)
    x0 = np.full(N, 5.0)
    if time.time() - start_time < DEADLINE:
        h_opt = _optimize_from(x0, max_iter=800)
        val_opt = _raw_value(h_opt)
        if val_opt < best_val:
            best_h = h_opt
            best_val = val_opt
    
    # 2. Try structured initializations
    # Alternating signs
    if time.time() - start_time < DEADLINE:
        x_alt = np.array([5.0 if i % 2 == 0 else -5.0 for i in range(N)])
        h_alt = _optimize_from(x_alt, max_iter=500)
        val_alt = _raw_value(h_alt)
        if val_alt < best_val:
            best_h = h_alt
            best_val = val_alt
    
    # Block initialization: half positive, half negative
    if time.time() - start_time < DEADLINE:
        x_block = np.concatenate([np.full(N//2, 5.0), np.full(N//2, -5.0)])
        h_block = _optimize_from(x_block, max_iter=500)
        val_block = _raw_value(h_block)
        if val_block < best_val:
            best_h = h_block
            best_val = val_block
    
    # 3. Random restarts
    np.random.seed(42)
    for trial in range(5):
        if time.time() - start_time > DEADLINE - 10:
            break
        x_rand = np.random.randn(N) * 2.0
        h_rand = _optimize_from(x_rand, max_iter=500)
        val_rand = _raw_value(h_rand)
        if val_rand < best_val:
            best_h = h_rand
            best_val = val_rand
    
    # 4. Refine the best solution with a final optimization using lower temperature
    if time.time() - start_time < DEADLINE - 5:
        x_best = np.arctanh(np.clip(best_h, -0.9999, 0.9999))
        
        def _refine_objective(x):
            return _smooth_objective(x, T=20.0)
        
        res_refine = minimize(
            _refine_objective,
            x_best,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        h_final = np.tanh(res_refine.x)
        val_final = _raw_value(h_final)
        
        if val_final < best_val:
            best_h = h_final
            best_val = val_final
    
    return best_h
# EVOLVE-BLOCK-END
