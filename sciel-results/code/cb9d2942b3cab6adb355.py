"""New acquisition function using Cost-Normalized Upper Confidence Bound (UCB)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Score(x) = (-mu(x) + beta * sigma(x)) / cost(x)
    
    For minimization, we maximize the optimistic bound of the improvement.
    We reformulate the minimization of f as maximization of -f.
    The UCB for -f is -mu + beta * sigma.
    
    This approach encourages exploration (high sigma) and exploitation (low mu).
    Normalizing by cost favors cheaper evaluations.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero or extreme values
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Exploration parameter
    # Beta controls the balance between exploitation and exploration.
    # A value of 1.0 is a standard starting point.
    beta = 1.0
    
    # Calculate UCB score for minimization
    # Maximizing this score corresponds to minimizing the optimistic lower bound of f.
    scores = (-mu + beta * sigma) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
