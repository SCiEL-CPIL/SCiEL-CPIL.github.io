"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB)
    adapted for minimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed objective value
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure numerical stability for sigma and cost
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # UCB parameter (exploration rate)
    # Tuned for 8D problem with cost constraint
    kappa = 2.0
    
    # Calculate UCB for minimization
    # We want to find points with low mu and high sigma.
    # The "regret" bound is (f_best - mu) + kappa * sigma
    # This represents the potential improvement over f_best, accounting for uncertainty.
    ucb_regret = (f_best - mu) + kappa * sigma_safe
    
    # Ensure non-negative scores (if mu is much lower than f_best, this is already positive)
    # If mu > f_best + kappa*sigma, ucb_regret can be negative, meaning no expected improvement.
    ucb_regret = np.maximum(ucb_regret, 0.0)
    
    # Cost-aware scoring
    scores = ucb_regret / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
