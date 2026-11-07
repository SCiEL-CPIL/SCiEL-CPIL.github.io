"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI).
    
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x)).
    
    We use a cost-effective version: Score = PI(x) / cost(x).
    
    This approach is distinct from EI and UCB. PI is bounded between 0 and 1,
    making it more stable in the presence of large uncertainties or extreme
    function values. Dividing by cost encourages the selection of cheaper
    evaluations, which is crucial for the cost-aware objective.
    
    To avoid division by zero and handle numerical stability, we add a small
    epsilon to sigma and cost.
    """
    # Probability of Improvement
    # z = (f_best - mu) / sigma
    # pi = norm.cdf(z)
    
    # Add epsilon to sigma to prevent division by zero
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Calculate PI
    pi = norm.cdf(z)
    
    # Cost-aware score: PI / cost
    # Cost is in [1, 2]
    score = pi / (cost + eps)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
