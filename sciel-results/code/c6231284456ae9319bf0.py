"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Internal state for adaptive UCB
_state = {
    'n_evals': 0,
    'kappa_history': [],
    'last_time': None
}


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses an adaptive UCB strategy that balances exploration and exploitation.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean of the GP at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation of the GP at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher means more promising.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases: if sigma is zero or near-zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Track number of evaluations
    _state['n_evals'] += 1
    n = _state['n_evals']
    
    # Adaptive kappa: start with high exploration, decrease over time
    # Use a schedule that starts aggressive and settles into exploitation
    # For Weierstrass (multimodal), we want strong early exploration
    if n <= 5:
        # Initial phase: heavy exploration
        kappa = 3.0
    elif n <= 10:
        kappa = 2.5
    elif n <= 15:
        kappa = 2.0
    elif n <= 20:
        kappa = 1.5
    elif n <= 25:
        kappa = 1.0
    else:
        # Final phase: exploitation
        kappa = 0.5
    
    # UCB: mu + kappa * sigma
    # Since we're minimizing, we want to find low values.
    # The upper confidence bound for minimization is: mu - kappa * sigma
    # But wait - the task says "higher meaning evaluate next"
    # For minimization, we want candidates that might have low f(x).
    # The probability that f(x) < f_best is high when mu is low and/or sigma is high.
    # 
    # Actually, for minimization:
    # - If mu is much lower than f_best, that's promising (exploitation)
    # - If sigma is high, there's uncertainty, so f(x) could be much lower (exploration)
    #
    # A common approach: use -mu + kappa * sigma or (f_best - mu) + kappa * sigma
    # Let's use: score = (f_best - mu) + kappa * sigma
    # This rewards both low mean (good f_best - mu) and high uncertainty (good sigma)
    
    scores = (f_best - mu) + kappa * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
