"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    mu, sigma: shape (2000,), GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: shape (2000,), evaluation cost for each candidate
    """
    # Time guard: if we've been running too long, return a safe fallback
    # The caller manages the overall loop; we just ensure we don't hang.
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standard normal CDF and PDF
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Quantile-LCB (10th percentile for minimization, i.e., beta ~ 1.28)
    # Score = -mu + beta * sigma
    # We blend EI and LCB to balance exploitation and exploration
    # Finding suggests Quantile-LCB is effective.
    # Let's try a hybrid: EI + 0.5 * LCB
    lcb_score = -mu + 1.5 * sigma_safe
    
    # Combine: Weighted sum
    # Normalize LCB to be comparable to EI? EI is in objective units. LCB is in objective units.
    # They are both "improvement-like" measures.
    # EI is the expected amount of improvement.
    # LCB is a lower bound on the function value (negated for maximization).
    # Actually, let's just use the Quantile-LCB as the primary driver, 
    # as per the "cell findings" which showed a jump from 0.86 to 1.34.
    
    scores = lcb_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
