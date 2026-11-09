"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_DEADLINE = time.time() + 290.0  # Safety margin before 300s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Combines Expected Improvement (EI) with a small exploration term to balance
    exploitation and exploration. The exploration term is scaled to be significant
    only when uncertainty is high, but not to dominate the EI signal.
    
    For minimization:
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    
    We add a small UCB-like term: kappa * sigma, but with a small kappa to avoid
    over-exploration.
    
    The final score is: EI + kappa * sigma, with kappa chosen to be small (e.g., 0.1)
    to provide a gentle nudge towards exploration without disrupting the EI signal.
    
    Numerical stability is ensured by clipping sigma to a minimum value.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a flat score to avoid breaking the loop, but prefer exploitation
        return -mu
    
    # Ensure sigma is positive to avoid division by zero or negative sqrt
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration term to encourage sampling in high-uncertainty regions
    kappa = 0.1
    exploration_term = kappa * sigma_safe
    
    # Final score: EI + exploration term
    scores = ei + exploration_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
