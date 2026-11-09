"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI) and
    Upper Confidence Bound (UCB) to balance exploitation and exploration.
    
    EI is strong for exploitation, while UCB encourages exploring regions
    with high uncertainty, which is crucial for finding the global minimum
    in multimodal functions like Styblinski-Tang.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Avoid numerical issues with extreme z values
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute UCB score for minimization
    # We want low mu and high sigma.
    # UCB = -mu + beta * sigma
    beta = 3.0  # Exploration parameter
    ucb_score = -mu + beta * sigma
    
    # 3. Combine EI and UCB
    # We weight them to ensure EI dominates but UCB provides exploration pressure.
    # The scales of EI and UCB might differ. EI is in objective units.
    # UCB is also in objective units.
    # A small weight for UCB prevents it from overriding a clear EI signal.
    weight_ucb = 0.1
    score = ei + weight_ucb * ucb_score
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
