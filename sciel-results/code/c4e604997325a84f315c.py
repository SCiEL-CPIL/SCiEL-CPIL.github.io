"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global variables for deadline tracking
_START_TIME = None
_DEADLINE_LIMIT = 325.0  # 5 seconds margin before the 330s kill switch

def _get_elapsed_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling with cost normalization.
    
    The score is computed as:
    - Sample a latent function value from the GP predictive distribution
    - Normalize by cost to prefer cheaper evaluations
    - Use a deterministic fallback if time is running out
    
    Parameters:
    mu (np.ndarray): GP posterior means, shape (N,)
    sigma (np.ndarray): GP posterior standard deviations, shape (N,)
    f_best (float): Best observed function value so far.
    cost (np.ndarray): Evaluation costs for each candidate, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,). Higher scores indicate
                candidates that are more likely to reduce the regret
                per unit cost.
    """
    # Deadline guard: If we are close to the time limit, return a simple, fast heuristic
    # to ensure we return valid scores and don't crash/timeout.
    # The heuristic: Maximize -mu / cost (exploit known bests, prefer cheap).
    if _get_elapsed_time() > _DEADLINE_LIMIT:
        mu_arr = np.asarray(mu, dtype=np.float64)
        cost_arr = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost_arr, 1e-8)
        safe_scores = -mu_arr / cost_safe
        safe_scores = np.where(np.isfinite(safe_scores), safe_scores, 0.0)
        return safe_scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-zero
    sigma_safe = np.maximum(sigma, 1e-8)
    # Ensure cost is non-zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # Thompson Sampling: sample a latent function value for each candidate
    # f_sample = mu + sigma * epsilon, where epsilon ~ N(0, 1)
    # We want to minimize f, so we maximize -f_sample / cost
    rng = np.random.default_rng()
    epsilon = rng.standard_normal(len(mu))
    f_sample = mu + sigma_safe * epsilon
    
    # Score: -f_sample / cost (higher is better, prefer low f and low cost)
    scores = -f_sample / cost_safe
    
    # Replace any non-finite values with 0.0 to ensure validity
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
