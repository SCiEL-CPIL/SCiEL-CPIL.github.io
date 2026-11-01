"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to minimize the upper bound on C3.
    
    The metric is: max_k |conv(h, h)[k]| / (sum_i |h_i|)^2
    
    We use gradient-based optimization with smooth approximations of max and abs.
    """
    
    def objective(x):
        h = np.tanh(x)  # Bounded in (-1, 1)
        conv = np.convolve(h, h, mode='full')
        
        # Smooth absolute value
        eps = 1e-10
        abs_conv = np.sqrt(conv**2 + eps)
        
        # Soft max with temperature
        T = 2.0
        # To avoid overflow, subtract max
        max_abs = np.max(abs_conv)
        log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
        log_max = T * log_sum_exp
        
        sum_abs_h = np.sum(np.sqrt(h**2 + eps))
        
        # Return the upper bound
        value = log_max / (sum_abs_h**2)
        return value
    
    # Use multiple random initializations to find better local minima
    best_value = np.inf
    best_x = None
    
    # First try the seed initialization (all ones)
    x0 = np.full(N, 5.0)  # tanh(5) ≈ 0.9999
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
    if res.fun < best_value:
        best_value = res.fun
        best_x = res.x
    
    # Try a few random initializations
    np.random.seed(42)
    for _ in range(5):
        x0 = np.random.randn(N) * 2.0
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-10})
        if res.fun < best_value:
            best_value = res.fun
            best_x = res.x
    
    h_opt = np.tanh(best_x)
    return h_opt

# EVOLVE-BLOCK-END
