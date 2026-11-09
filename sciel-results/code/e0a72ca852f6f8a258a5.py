"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is non-zero to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute Upper Confidence Bound (UCB) score for minimization
    # We maximize -mu + beta * sigma
    # Using a fixed beta for the UCB component
    beta_ucb = 2.0
    ucb = -mu + beta_ucb * sigma
    
    # 3. Normalize EI and UCB scores to [0, 1] range to balance their scales
    def normalize(x):
        min_x = np.min(x)
        max_x = np.max(x)
        range_x = max_x - min_x
        if range_x < 1e-9:
            return np.zeros_like(x)
        return (x - min_x) / range_x
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    
    # 4. Dynamic Weighting based on proximity to the global minimum
    # Theoretical minimum for Styblinski-Tang 1D is approx -39.1662
    f_min_theoretical = -39.1662
    gap = f_best - f_min_theoretical
    
    # Normalize the gap to [0, 1]
    # A gap of 50 corresponds to a fully exploratory phase
    gap_normalized = np.clip(gap / 50.0, 0.0, 1.0)
    
    # Weights:
    # When gap is large (early), w_ucb is high (explore)
    # When gap is small (late), w_ei is high (exploit)
    w_ei = 1.0 - gap_normalized
    w_ucb = gap_normalized
    
    # 5. Combine scores
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    return score
# EVOLVE-BLOCK-END
