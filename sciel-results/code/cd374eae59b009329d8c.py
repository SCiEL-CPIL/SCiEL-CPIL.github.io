"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Use a fixed deadline guard: if we exceed 330s wall-clock, we must return.
# Since this function is called many times, we track a global start time.
# However, the contract says "Each evaluation must finish within 300 s".
# The evaluator KILLS the program after 330s wall-clock for the whole run?
# "The evaluator measures... Keep the function names... Each evaluation must finish within 300 s."
# "the evaluator KILLS the program after 330s wall-clock."
# This implies the *entire run* (all iterations) must finish in 330s.
# So we should track elapsed time globally.

_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, lowest observed value.
        cost: (2000,) array of evaluation costs.
    
    Returns:
        (2000,) array of finite scores (higher = evaluate next).
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # Return a safe fallback: just pick the one with lowest mu / cost
        # to avoid crashing, though the run is likely over.
        return -mu / (cost + 1e-9)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Strategy: Expected Improvement (EI) per unit cost, with a slight exploration bonus.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Compute z
    z = (f_best - mu) / sigma_safe
    
    # Use scipy.stats.norm for cdf and pdf
    from scipy.stats import norm
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Cost-aware: divide by cost
    score = ei / cost_safe
    
    # Add a small exploration term to break ties and encourage diversity early on?
    # The current seed program is just EI/cost.
    # Let's try a hybrid: EI/cost + small * sigma/cost
    # Or maybe just EI/cost is fine, but let's ensure numerical stability.
    
    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return score
# EVOLVE-BLOCK-END
