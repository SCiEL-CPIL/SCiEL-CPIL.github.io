import numpy as np
import time

# Global deadline guard
_START_TIME = None
_DEADLINE = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Parameters:
    mu: array of shape (2000,), predicted mean from GP
    sigma: array of shape (2000,), predicted standard deviation from GP
    f_best: float, best function value observed so far
    cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
    scores: array of shape (2000,), acquisition scores (higher is better)
    """
    global _START_TIME, _DEADLINE
    
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 280.0  # 50s safety margin before 330s limit
    
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple fallback: favor low mu, high sigma, low cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        scores = -mu / np.sqrt(cost_safe)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample from posterior predictive N(mu, sigma)
    # Use a fixed seed per call based on iteration count for reproducibility within a run
    # But we don't have iteration count, so use a random seed each time
    rng = np.random.default_rng()
    
    # Draw samples from the posterior
    samples = mu + sigma_safe * rng.standard_normal(n)
    
    # For minimization, lower sample values are better
    # Thompson sampling score: negative of the sample, adjusted for cost
    # We want higher scores for better (lower) function values
    # Normalize by cost to make it cost-aware
    
    # Option 1: score = -sample / cost
    # Option 2: score = -sample / sqrt(cost)
    # Option 3: score = -sample / log(1 + cost)
    
    # Let's use a hybrid: penalize cost but not too harshly
    # Also add a small exploration bonus for high uncertainty
    exploration_bonus = sigma_safe * 0.1
    
    scores = (-samples + exploration_bonus) / np.sqrt(cost_safe)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
