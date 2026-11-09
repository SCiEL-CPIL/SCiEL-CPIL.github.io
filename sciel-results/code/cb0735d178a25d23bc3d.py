# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s. We use a safe margin.
_DEADLINE = time.time() + 320.0  # 320s to be safe

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe default: exploit best
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Knowledge Gradient (KG) acquisition function
    # KG maximizes the expected reduction in the minimum observed value
    # E[max(0, f_best - Y)] where Y ~ N(mu, sigma^2)
    # This is equivalent to:
    # (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # which is actually the same as Expected Improvement (EI)
    #
    # For a proper KG that accounts for the fact that we only evaluate one point,
    # we use the standard EI formula but with a cost-aware weighting.
    #
    # The key insight for KG is that it should prioritize points that are likely
    # to significantly reduce f_best. We use a modified approach:
    # - Compute EI for each point
    # - Weight by the probability that the point is actually better than f_best
    # - Divide by cost to account for the evaluation cost
    
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of improvement (how likely this point is to beat f_best)
    p_improve = norm.cdf(z)
    
    # Knowledge Gradient: weight EI by the probability of improvement
    # This emphasizes points that are both likely to improve AND have high expected improvement
    kg = ei * p_improve
    
    # Cost-aware: divide by cost
    scores = kg / cost
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local minima and explore the space
    # The exploration bonus is weighted by the inverse cost
    exploration_bonus = 0.02 * sigma / cost
    
    scores = scores + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
