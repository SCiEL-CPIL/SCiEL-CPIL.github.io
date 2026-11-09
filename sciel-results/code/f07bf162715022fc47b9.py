"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to minimize f(x).
    # Lower Confidence Bound LCB(x) = mu(x) - beta * sigma(x).
    # We select the point with the lowest LCB.
    # The evaluator selects the point with the HIGHEST score.
    # So we return Score(x) = -LCB(x) = beta * sigma(x) - mu(x).
    
    beta = 1.0
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    scores = beta * sigma - mu
    
    # Handle any potential NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
