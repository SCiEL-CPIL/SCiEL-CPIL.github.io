import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Adaptive UCB (Lower Confidence Bound for minimization, so we maximize the negative)
    # Score = -mu + beta * sigma
    
    # Calculate adaptive beta
    # Use f_best as a proxy for progress. 
    # When f_best is large, we are far from optimum, so increase exploration (beta).
    # When f_best is small, we are close, so decrease exploration.
    
    # Ensure numerical stability
    fb = max(float(f_best), 1e-8)
    
    # Heuristic: Beta scales with the log of the current best value
    # The constant 2.0 provides a baseline level of exploration
    beta = 2.0 + np.sqrt(2.0 * np.log(fb + 1.0))
    
    # Compute scores
    scores = -mu + beta * sigma
    
    return scores
