"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Cost-normalized UCB with high gamma.
    
    Score = (mu + kappa * sigma) / cost^gamma
    
    This aggressively explores cheap, uncertain regions of the domain,
    which is critical for locating global minima within a tight budget.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    score : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    t0 = time.time()
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Cost-normalized UCB
    # kappa controls exploration vs exploitation
    # gamma controls how aggressively we penalize high-cost points
    kappa = 1.5
    gamma = 3.0
    
    # Compute UCB: mu + kappa * sigma
    # We want to maximize this, but also account for f_best
    # Actually, for minimization, we want to minimize, so we use:
    # score = (f_best - mu + kappa * sigma) / cost^gamma
    # But this can be negative. Let's think differently.
    
    # For minimization, we want points where mu is low and sigma is high.
    # UCB for minimization: mu - kappa * sigma (lower is better)
    # But we want higher scores to be better for acquisition.
    # So: score = (f_best - (mu - kappa * sigma)) / cost^gamma
    #      = (f_best - mu + kappa * sigma) / cost^gamma
    
    # This is similar to EI but with a linear sigma term instead of the
    # full EI formula. It's more aggressive in exploration.
    
    # Compute the numerator: f_best - mu + kappa * sigma
    numerator = f_best - mu + kappa * sigma_safe
    
    # Ensure non-negative (points with mu > f_best + kappa*sigma get 0)
    numerator = np.maximum(numerator, 0.0)
    
    # Compute cost penalty
    cost_safe = np.maximum(cost, 1e-12)
    cost_penalty = cost_safe ** gamma
    
    # Final score
    score = numerator / cost_penalty
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - t0
    if elapsed > 250:  # Safety margin
        # Return a simple fallback if we're running too long
        pass
    
    return score
    # EVOLVE-BLOCK-END
