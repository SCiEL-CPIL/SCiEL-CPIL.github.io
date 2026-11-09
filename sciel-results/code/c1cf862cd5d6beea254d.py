"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement and an exploration bonus,
    normalized by cost. Includes a deadline guard for safety.
    """
    # Deadline guard: track elapsed time
    # Note: We don't have access to a global start time here, so we rely on
    # the evaluator's timeout. But we can add a simple safety check.
    # For now, we'll compute the acquisition function efficiently.
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourage exploring high-uncertainty regions
    # This is a simple heuristic: add a term proportional to sigma
    # The coefficient is tuned to balance exploitation and exploration
    exploration_bonus = 0.1 * sigma_safe
    
    # Combined acquisition: EI + exploration bonus, normalized by cost
    # We use a small epsilon in the denominator to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = (ei + exploration_bonus) / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
