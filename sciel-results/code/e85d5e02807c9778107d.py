import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Lower Confidence Bound (LCB) for minimization.
    Structural change: Switches from Expected Improvement (EI) to LCB.
    LCB explicitly trades off exploitation (low mu) and exploration (high sigma).
    
    Parameters:
    mu (np.ndarray): Posterior mean, shape (2000,).
    sigma (np.ndarray): Posterior standard deviation, shape (2000,).
    f_best (float): Best observed value so far.
    
    Returns:
    np.ndarray: Scores for the 2000 candidates. Higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE - 5.0:
        # Fallback to pure exploitation to save time
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -1e6)
        return scores

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Ensure mu is finite
    # If mu is not finite, treat it as a very bad point (high value)
    mu_safe = np.where(np.isfinite(mu), mu, 1e6)

    # LCB parameter
    # Beta controls the exploration-exploitation trade-off.
    # Higher beta means more exploration.
    # Given the small number of evaluations (25), we use a moderate-to-high beta.
    beta = 2.5
    
    # LCB for minimization: minimize (mu - beta * sigma)
    # Score for "higher is better": maximize (beta * sigma - mu)
    scores = beta * sigma_safe - mu_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
