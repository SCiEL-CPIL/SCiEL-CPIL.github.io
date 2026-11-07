"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a Bayesian model evidence (negative log-likelihood)
    criterion combined with cost efficiency.
    
    The score is based on:
    - Negative log-likelihood approximation: -(f_best - mu)/sigma - log(sigma)
      This balances exploitation (low mu relative to f_best) against uncertainty
      (penalizing high sigma via the log term).
    - Inverse cost weighting to prefer cheaper evaluations.
    
    We combine these into a single score that favors points with low predicted
    values, moderate uncertainty, and low cost.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute the negative log-likelihood proxy
    # -(f_best - mu) / sigma - log(sigma)
    # This is positive when mu < f_best (improvement region) and penalizes high sigma
    nll_proxy = -(f_best - mu) / sigma_safe - np.log(sigma_safe)
    
    # Compute a cost-aware score: nll_proxy / cost
    # We want higher scores to be better
    scores = nll_proxy / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
