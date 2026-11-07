"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard: kill long-running computations safely.
_T0 = time.time()
_HARD_DEADLINE = 315.0  # seconds; leave ~15s buffer for the harness


def _elapsed_ok():
    return (time.time() - _T0) < _HARD_DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = better to evaluate next).
    
    Strategy: Expected Improvement per unit cost.
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    Score = EI(x) / cost(x)
    """
    if not _elapsed_ok():
        # Emergency: return something finite
        return np.zeros_like(mu)
    
    # Guard against zero or near-zero sigma
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Where sigma is effectively zero, EI is just max(0, f_best - mu)
    ei = np.where(sigma > 1e-12, ei, np.maximum(0.0, f_best - mu))
    
    # Cost-aware EI
    scores = ei / cost_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
