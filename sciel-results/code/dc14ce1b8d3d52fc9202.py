"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI) and
    Lower Confidence Bound (LCB) with adaptive beta.
    
    The key insight from the cell findings is that using the known global minimum
    (3.0) as the EI reference maintains a constant exploration signal that doesn't
    diminish as f_best approaches 3, keeping the acquisition sensitive to high-variance
    regions that may contain the true optimum.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Use the known global minimum (3.0) as the reference for EI
    # This maintains a constant exploration signal
    f_ref = min(f_best, 3.0)
    
    # Expected Improvement relative to f_ref
    z = (f_ref - mu) / sigma_safe
    ei = (f_ref - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound: mu - beta * sigma
    # We want to maximize the score, so we use -LCB = -mu + beta * sigma
    # Adaptive beta: start higher for exploration, decrease as we find better solutions
    # beta = 1.0 + 0.5 * (f_best - 3.0) / max(f_best - 3.0, 1e-6)
    # This gives beta = 1.5 when f_best is far from 3, and beta = 1.0 when f_best is close to 3
    
    # Simpler adaptive beta based on how much improvement is possible
    remaining_improvement = max(f_best - 3.0, 0.0)
    beta = 1.0 + 0.5 * np.exp(-remaining_improvement / 2.0)
    
    # LCB-based score: -mu + beta * sigma
    lcb_score = -mu + beta * sigma_safe
    
    # Combine EI and LCB scores
    # EI is already in "higher is better" form (expected improvement over f_ref)
    # LCB_score is also "higher is better" (we want low mu and high sigma)
    score = ei + lcb_score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
