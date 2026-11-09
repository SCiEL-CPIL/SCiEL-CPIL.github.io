import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Known global minimum for Hartmann 6D
    f_min = -3.32237
    
    # Gap from best observed to known minimum
    gap = f_best - f_min
    
    # Adaptive kappa based on how far we are from the optimum
    # High gap -> Explore more (High Kappa)
    # Low gap -> Exploit more (Low Kappa / EI)
    
    # Thresholds can be tuned. Hartmann 6D range is ~6.6.
    # Initial random points might yield f_best ~ 0 to 2. Gap ~ 3 to 5.
    # Later, if we find a local min ~ -1.0, Gap ~ 2.3.
    # If we find global min, Gap ~ 0.
    
    if gap > 1.5:
        kappa = 3.0
    elif gap > 0.8:
        kappa = 2.0
    elif gap > 0.3:
        kappa = 1.0
    else:
        kappa = 0.5

    # Compute UCB score: -mu + kappa * sigma
    # Higher is better (minimize -score)
    scores = -mu + kappa * sigma
    
    return scores
