"""Cost-aware acquisition using Knowledge Gradient approximation for Griewank 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Knowledge Gradient (KG) approximation
    for Griewank 2D Bayesian Optimization.
    
    KG approximates the expected reduction in the optimal value estimate.
    For minimization, this is modeled as the probability that a candidate
    improves upon f_best weighted by the magnitude of improvement.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive and finite
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Knowledge Gradient approximation for minimization:
    # Expected improvement = (f_best - mu) * (1 - Phi((f_best - mu) / sigma))
    # This captures both the probability of improvement and the expected magnitude.
    improvement = f_best - mu
    z = improvement / sigma_safe
    # Use survival function for numerical stability
    sf = norm.sf(z)
    kg_score = improvement * sf
    
    # Clip to non-negative (KG should be non-negative)
    kg_score = np.maximum(kg_score, 0.0)
    
    # Cost-aware normalization
    # Divide by cost to penalize expensive evaluations.
    score = kg_score / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
