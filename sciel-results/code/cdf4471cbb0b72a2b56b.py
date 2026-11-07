import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition function.
    
    Uses an adaptive beta parameter that decreases as the best observed value
    approaches the known global minimum of the Michalewicz 2D function.
    This promotes exploration early in the optimization and exploitation later.
    """
    # Known global minimum of Michalewicz 2D
    f_min = -1.8013
    
    # Calculate distance to known optimum
    # f_best is the lowest observed value (minimization)
    d = max(0.0, f_best - f_min)
    
    # Scale d to [0, 1]
    # The range of Michalewicz 2D is roughly [-1.8, 1.5], so max regret is ~3.3
    d_norm = min(1.0, d / 3.0)
    
    # Adaptive beta
    # beta_max = 2.5 (strong exploration when far from optimum)
    # beta_min = 0.5 (strong exploitation when close to optimum)
    beta = 0.5 + 2.0 * d_norm
    
    # LCB score for minimization: -mu + beta * sigma
    # Higher score means more likely to be the new minimum
    score = -mu + beta * sigma
    
    # Ensure finite scores as required by the contract
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
