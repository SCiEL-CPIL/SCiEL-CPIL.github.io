# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill

# Global state to track number of evaluations (approximate)
_eval_count = [0]


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using GP-UCB.
    
    Uses Upper Confidence Bound with a time-varying exploration parameter
    beta_t = sqrt(log(t + 1)) where t is approximated from the number of
    evaluations. This systematically balances exploitation of low-mean
    regions against exploration of high-uncertainty areas.
    
    Parameters:
        mu: array of shape (2000,), posterior mean at candidates
        sigma: array of shape (2000,), posterior std at candidates
        f_best: scalar, best (lowest) observed value so far
    
    Returns:
        array of shape (2000,), scores (higher = better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a simple fallback
        return -mu + 0.1 * sigma
    
    # Approximate evaluation count from f_best magnitude
    # Early on, f_best is likely far from optimum; later it gets closer
    # We can estimate t by tracking how many times we've been called
    _eval_count[0] += 1
    t = _eval_count[0]
    
    # Time-varying exploration parameter
    # beta_t = sqrt(log(t + 1)) gives a schedule that starts high and decays
    # But for this problem, we want to maintain reasonable exploration
    # Use a modified schedule: beta = min(sqrt(log(t + 1)), max_beta)
    # with a floor to ensure some exploration always
    beta = np.sqrt(np.log(t + 1))
    
    # Scale beta appropriately for the problem
    # The sigma values are in objective units for an RBF mixture
    # with minimum around -2.5, so typical sigma might be 0.1-1.0
    # We want beta * sigma to be meaningful relative to f_best - mu
    # Start with a moderate beta and let it evolve
    beta = max(beta, 1.0)  # Ensure minimum exploration
    beta = min(beta, 3.0)  # Cap to avoid excessive exploration
    
    # Guard against division by zero / numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # UCB score for minimization: we want to maximize (minimize -mu + beta*sigma)
    # Actually, for minimization, UCB is: score = -mu + beta * sigma
    # Higher score = more promising (lower expected value + uncertainty bonus)
    score = -mu + beta * sigma_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
