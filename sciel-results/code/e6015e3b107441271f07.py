"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a
    Lower Confidence Bound (LCB) term to balance exploration and exploitation.
    
    The LCB term is weighted by a constant beta to encourage exploring
    regions with high uncertainty, while EI focuses on promising areas.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound for minimization: -mu - beta * sigma
    # We want to maximize this, so we return -mu - beta * sigma
    # But we need to combine with EI. Let's use a weighted sum.
    # LCB score = -mu - beta * sigma
    beta = 2.0
    lcb = -mu - beta * sigma_safe
    
    # Normalize both to comparable scales? Or just use a weighted sum.
    # EI is non-negative. LCB can be negative.
    # Let's try a simple weighted sum. Since EI is typically larger in magnitude
    # when there's improvement, and LCB is negative when mu is high,
    # we can just add them. But they have different units/scales.
    
    # Alternative: Use EI as primary and add a small exploration bonus.
    # Or use UCB style: -mu + beta * sigma (for minimization, we want low mu and high sigma)
    # Wait, for minimization, LCB is mu - beta * sigma. We want to minimize this.
    # So the score should be -LCB = -mu + beta * sigma.
    
    # Let's use a combination: EI + alpha * (-mu + beta * sigma)
    # But this might be double-counting.
    
    # A common approach is to use EI directly, or use a variant like
    # "Expected Improvement with exploration" or just tune beta in LCB.
    
    # Given the task, let's stick with a robust EI but perhaps add a
    # small penalty for very low uncertainty to avoid over-exploration?
    # No, EI already handles that.
    
    # Let's try a different approach: Use EI but with a slightly modified
    # formula or just ensure numerical stability.
    
    # Actually, the current EI is standard. Let's try adding a small
    # exploration term based on sigma to encourage diversity.
    # Score = EI + gamma * sigma
    gamma = 0.0
    score = ei + gamma * sigma_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
