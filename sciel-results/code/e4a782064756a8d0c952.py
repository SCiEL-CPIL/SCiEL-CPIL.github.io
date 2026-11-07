"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Thompson Sampling: sample a single objective value from the Gaussian
    posterior for each candidate, normalized by evaluation cost.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Thompson sampling: draw one sample from the posterior for each candidate
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(mu.shape)
    
    # For minimization, lower sampled values are better.
    # We want higher scores to be evaluated next, so invert: score = -sample / cost
    # But we also want to prefer points that are likely better than f_best.
    # Use: score = (f_best - sample) / cost, clipped to non-negative
    scores = (f_best - samples) / cost_safe
    
    # Clip to non-negative: only consider candidates that might beat f_best
    scores = np.clip(scores, 0.0, None)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
