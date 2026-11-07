"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    We want to find the candidate with the highest potential to be a new minimum.
    The optimistic estimate of the function value at x is mu(x) - beta * sigma(x).
    We want to maximize the "negativity" of this estimate, i.e., minimize (mu - beta * sigma).
    So the score is - (mu - beta * sigma) = -mu + beta * sigma.
    
    Higher score means we should evaluate this candidate next.
    """
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Beta controls exploration vs exploitation.
    # A value of 2.5 is a common starting point for exploration in multi-modal landscapes.
    beta = 2.5
    
    # Score: -mu + beta * sigma
    # -mu favors low mean (exploitation)
    # beta * sigma favors high uncertainty (exploration)
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
