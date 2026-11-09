"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI)
    with a small epsilon, normalized by cost.
    
    PI is often more robust than EI for sparse data because it focuses
    on finding *any* improvement rather than maximizing expected gain,
    which can be overly conservative in early stages.
    
    We use:
        score = P(x < f_best - eps) / cost
              = Phi((f_best - eps - mu) / sigma) / cost
    
    This encourages exploration of regions where the probability of
    beating the best (with a small margin) is high, weighted by cost.
    """
    eps = 1e-3  # Small margin to avoid pure exploitation of noise
    z = (f_best - eps - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    # Ensure finite scores
    scores = pi / (cost + 1e-9)
    # Replace any non-finite values with 0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
