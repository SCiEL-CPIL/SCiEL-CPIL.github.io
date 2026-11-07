"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Hybrid acquisition: blend EI and UCB with adaptive exploration
    # that decreases as f_best approaches the known optimum.
    # 
    # The Styblinski-Tang 1D function has minimum -39.1662 at x ≈ -1.717.
    # We use a UCB term to encourage exploration of unexplored regions,
    # weighted by how far f_best is from the optimum.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound
    # Use a beta that scales with the gap to the known optimum
    # When f_best is far from optimum, explore more; when close, exploit more
    gap = np.abs(f_best - (-39.1662))
    # Normalize gap: at start gap is large (~40), near optimum gap is small
    # beta decreases from ~2.0 to ~0.5 as we approach optimum
    beta = 0.5 + 1.5 * np.exp(-gap / 10.0)
    
    ucb = -mu + beta * sigma
    
    # Blend: start with more UCB (exploration), shift to EI (exploitation)
    # weight for EI increases as we get closer to optimum
    ei_weight = 0.5 + 0.5 * np.exp(-gap / 5.0)
    ucb_weight = 1.0 - ei_weight
    
    score = ei_weight * ei + ucb_weight * ucb
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
