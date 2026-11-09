"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guarding
_START_TIME = time.time()
# Safety margin in seconds (330s limit, leave 20s buffer)
_DEADLINE_MARGIN = 310.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement with adaptive exploration.
    
    Combines pure EI (which scored well) with a small, adaptive uncertainty
    bonus to encourage exploration in early stages or when uncertainty is high.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition values; higher means evaluate next.
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_MARGIN:
        return -mu

    # Ensure sigma is non-negative and finite
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma^2)
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z_clipped = np.clip(z, -8.0, 8.0)
    
    # Compute EI using clipped z for stability
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Adaptive exploration bonus:
    # Add a small term proportional to sigma, scaled by how much uncertainty
    # remains relative to the current improvement potential.
    # This helps escape local optima in the multimodal Weierstrass function.
    
    # Measure the scale of improvement
    improvement = f_best - mu
    improvement_scale = np.mean(np.abs(improvement))
    
    if improvement_scale > 1e-9:
        # Uncertainty ratio: how much sigma is relative to typical improvement
        uncertainty_ratio = np.mean(sigma) / improvement_scale
    else:
        uncertainty_ratio = 1.0
    
    # Clip uncertainty ratio to a reasonable range
    uncertainty_ratio = np.clip(uncertainty_ratio, 0.1, 5.0)
    
    # Small exploration weight: 0.1 to 0.3 depending on uncertainty
    # This keeps EI dominant but adds a gentle push toward uncertain regions
    w_explore = 0.1 + 0.2 * min(uncertainty_ratio, 2.0) / 2.0
    
    # Add exploration bonus
    scores = ei + w_explore * sigma
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
