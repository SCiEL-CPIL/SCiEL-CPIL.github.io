"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Args:
        mu: Mean of the GP posterior, shape (N,).
        sigma: Standard deviation of the GP posterior, shape (N,).
        f_best: Best observed function value.
        cost: Cost of evaluating each candidate, shape (N,).
        
    Returns:
        scores: Acquisition scores, shape (N,). Higher is better.
    """
    # UCB for minimization: we want to maximize the potential to find low values.
    # The lower bound on f(x) is mu - kappa * sigma.
    # We want to maximize - (mu - kappa * sigma) = -mu + kappa * sigma.
    
    # Choose kappa. 
    # A fixed kappa of 2.0 is a reasonable default for 2D multimodal functions 
    # like Rastrigin, balancing exploration and exploitation.
    kappa = 2.0
    
    # Calculate UCB scores
    ucb_scores = -mu + kappa * sigma
    
    # Cost-aware adjustment: divide by cost.
    # This penalizes expensive points, encouraging the search to stay in 
    # cheaper regions if the function values are comparable.
    # Cost is guaranteed to be >= 1.0 in this problem.
    final_scores = ucb_scores / cost
    
    # Ensure all scores are finite to avoid invalidating the result.
    # Replace NaN or Inf with a very low score.
    final_scores = np.where(np.isfinite(final_scores), final_scores, -1e10)
    
    return final_scores
# EVOLVE-BLOCK-END
