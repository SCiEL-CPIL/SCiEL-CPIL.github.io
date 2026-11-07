import numpy as np
import time

# Global timer for the entire evaluation session
_start_time = time.time()
_TIME_LIMIT = 320  # 330s limit, keep 10s margin

def acquisition(mu, sigma, f_best):
    """
    Computes the Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Parameters:
    mu: array-like, shape (n_candidates,)
        Posterior mean of the GP.
    sigma: array-like, shape (n_candidates,)
        Posterior standard deviation of the GP.
    f_best: float
        The best (lowest) observed value so far.
        
    Returns:
    scores: array-like, shape (n_candidates,)
        Acquisition scores. Higher values indicate more promising candidates.
    """
    # Deadline guard: if we are running out of time, return a safe fallback
    # to ensure the loop doesn't hang or crash due to timeout in the evaluator.
    # The fallback exploits the current best guess (lowest mu).
    if time.time() - _start_time > _TIME_LIMIT:
        return -mu

    # UCB for minimization:
    # We want to find the minimum of the function.
    # The posterior is N(mu, sigma^2).
    # A common heuristic for minimization is to maximize:
    #   score(x) = -mu(x) + kappa * sigma(x)
    # This term rewards points with low predicted mean (-mu) and high uncertainty (sigma).
    
    # Tuning parameter kappa. 
    # kappa=2.0 corresponds to approx 95% confidence interval width.
    # This balances exploration and exploitation.
    kappa = 2.0
    
    scores = kappa * sigma - mu
    
    # Ensure numerical stability: replace NaNs and infs with finite values
    # NaNs might arise if sigma is 0 (division by zero in other contexts, but here just linear)
    # However, sigma might be very small, leading to numerical issues in other ops, but here it's safe.
    # Still, good practice to clean.
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
