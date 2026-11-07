"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB for Bayesian optimization.
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard: track elapsed time
    t0 = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero and negative values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) with beta tuned for exploration
    beta = 2.0
    ucb = mu + beta * sigma_safe
    
    # Combine EI and UCB: use EI for exploitation, UCB for exploration
    # Normalize both to comparable scales
    # EI is typically positive when mu < f_best, UCB rewards high uncertainty
    # We want higher scores for better candidates
    
    # Cost-normalized combination
    # EI per unit cost
    ei_cost = ei / (cost + 1e-9)
    
    # UCB per unit cost (note: lower UCB is better for minimization, so we negate)
    # Actually, for minimization, we want low mu and high sigma for exploration
    # UCB = mu + beta*sigma gives an upper bound on the true value
    # For minimization, we prefer low UCB, so we should use -UCB or (f_best - UCB)
    
    # Let's use a hybrid: weight EI heavily but add exploration bonus
    # The key insight: when sigma is high, we should explore more
    # When sigma is low and mu < f_best, we should exploit
    
    # Simple approach: EI + exploration_bonus - exploitation_penalty
    # But let's stick with a clean combination
    
    # For minimization:
    # - High EI means likely improvement
    # - High sigma means uncertain, worth exploring
    # - Low cost is preferred
    
    # Combined score: (EI + gamma * sigma) / cost
    # where gamma controls exploration vs exploitation
    
    gamma = 0.5  # Exploration weight
    
    score = (ei + gamma * sigma_safe) / (cost + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - t0 > 250:
        return score
    
    return score
# EVOLVE-BLOCK-END
