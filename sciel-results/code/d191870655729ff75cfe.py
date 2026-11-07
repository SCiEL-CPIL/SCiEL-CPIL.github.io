"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Combine Expected Improvement and UCB for balanced exploration/exploitation.
    # EI exploits known good regions, UCB explores uncertain areas.
    eps = 1e-9
    sigma_safe = sigma + eps
    
    # Expected Improvement
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # UCB (Upper Confidence Bound) for minimization
    # We want to explore where uncertainty is high, so we use -mu + beta*sigma
    # But since we're minimizing, we want to find low values, so we use mu - beta*sigma
    # However, the acquisition score should be higher for better candidates
    # So for minimization, we want high scores for low mu and high sigma
    # UCB for minimization: score = -mu + beta * sigma
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Normalize both to have similar scale
    # EI is in objective units, UCB is also in objective units
    # Weighted combination
    w_ei = 0.7
    w_ucb = 0.3
    
    score = w_ei * ei + w_ucb * ucb
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
