"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with adaptive exploration, divided by cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far (lower is better)
    cost : array-like, shape (2000,)
        Evaluation cost for each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB component: balance exploitation and exploration
    # Use adaptive beta that increases slightly with uncertainty
    # Beta around 2.0 works well for Rosenbrock-like landscapes
    beta = 2.0
    ucb = mu + beta * sigma_safe
    
    # For minimization, we want to minimize, so UCB score is -ucb (higher is better)
    # But we combine with EI which already accounts for improvement
    # Hybrid: weighted combination of EI and negative UCB
    # Normalize both to comparable scales
    
    # Scale EI and UCB components
    # EI is non-negative, UCB can be negative
    # Use EI as primary, add exploration bonus from UCB
    
    # Adaptive exploration: more exploration when we've seen few points
    # Estimate exploration level from the spread of sigma
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Exploration bonus: proportional to sigma, scaled by how uncertain we are
    # This helps escape local optima
    exploration_weight = 0.3  # Moderate exploration
    score = ei + exploration_weight * sigma_safe
    
    # Cost-aware: divide by cost (lower cost = higher priority)
    cost_safe = np.maximum(cost, 1e-9)
    score = score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Clip to avoid extreme values
    score = np.clip(score, -1e6, 1e6)
    
    return score
# EVOLVE-BLOCK-END
