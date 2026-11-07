"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses a combination of Expected Improvement (EI) and an exploration term
    based on the posterior standard deviation to balance exploitation and exploration.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Exploration term: we want to explore areas with high uncertainty
    # but also consider how much improvement is possible.
    # A common approach is to use a weighted sum of EI and sigma, or to use
    # a variant like UCB (Upper Confidence Bound) for maximization, which
    # translates to LCB (Lower Confidence Bound) for minimization.
    # For minimization, LCB = mu - beta * sigma.
    # We want to maximize the acquisition score, so we can use -LCB = beta * sigma - mu.
    # However, EI already incorporates some exploration.
    # Let's try a simple hybrid: EI + alpha * sigma, where alpha is a small constant
    # to encourage exploration in uncertain regions.
    # Alternatively, we can use a more principled approach like PI (Probability of Improvement)
    # combined with EI.
    
    # Let's use EI as the primary term and add a small exploration bonus.
    # The exploration bonus should be scaled to be comparable to EI.
    # Since EI is in objective units and sigma is also in objective units,
    # we can add a small multiple of sigma.
    
    # Let's try: scores = EI + 0.1 * sigma
    # This encourages exploration in regions with high uncertainty but doesn't
    # completely dominate the exploitation signal from EI.
    
    exploration_weight = 0.1
    scores = ei + exploration_weight * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
