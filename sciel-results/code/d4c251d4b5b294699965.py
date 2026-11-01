"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _c3_upper_bound(h):
    """
    Compute the upper bound on C3 for a signed step function with heights h.
    Metric: max_k |(h*h)_k| / (sum_i |h_i|)^2
    """
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(conv))
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h == 0:
        return float('inf')
    return max_conv / (sum_abs_h ** 2)


def _smooth_c3_objective(x, T=10.0):
    """
    Smooth approximation of the C3 upper bound for optimization.
    x are the parameters, h = tanh(x).
    """
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    abs_conv = np.abs(conv)
    
    # Smooth max using log-sum-exp
    # To prevent overflow, subtract max
    if len(abs_conv) > 0:
        max_val = np.max(abs_conv)
        if max_val > 0:
            # T controls sharpness of max. Larger T = softer max.
            # We want the gradient to push down the largest peaks.
            # log(sum(exp((a_i - max)/T))) * T + max
            # This is approx max(a_i)
            log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_val) / T)))
            smooth_max = T * log_sum_exp + max_val
        else:
            smooth_max = 0.0
    else:
        smooth_max = 0.0

    # Smooth L1 norm: sum(sqrt(h_i^2 + eps)) approximates sum(|h_i|)
    eps = 1e-6
    sum_abs_h = np.sum(np.sqrt(h**2 + eps))
    
    if sum_abs_h < 1e-6:
        return 1e6
    
    return smooth_max / (sum_abs_h ** 2)


def construct():
    # We want to minimize the C3 upper bound.
    # The seed is all ones, which gives a high bound.
    # Random signs give a much lower bound.
    # We will try several random initializations and pick the best.
    
    best_h = None
    best_val = float('inf')
    
    # Try a few random starts
    rng = np.random.RandomState(42)
    
    for i in range(5):
        # Initialize with random signs smoothed to avoid exactly 0 or 1
        # tanh(x) = sign, so x should be large positive or negative
        # Random signs: x_i ~ Uniform[-5, 5] gives h_i in [-0.9999, 0.9999]
        x0 = rng.uniform(-5, 5, N)
        
        # Optimize
        try:
            res = minimize(
                _smooth_c3_objective, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            h_opt = np.tanh(res.x)
            val = _c3_upper_bound(h_opt)
            
            if val < best_val:
                best_val = val
                best_h = h_opt
        except Exception:
            pass
            
    # If optimization failed or didn't improve, fall back to a simple random sign pattern
    if best_h is None or best_val > 0.001: # 0.001 is much better than seed 0.0025
        # Use a fixed random pattern that is known to have low autocorrelation
        # e.g. m-sequence or just random
        x0 = rng.uniform(-10, 10, N)
        best_h = np.tanh(x0)
        best_val = _c3_upper_bound(best_h)

    return best_h

# EVOLVE-BLOCK-END
