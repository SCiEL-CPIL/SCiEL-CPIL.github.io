"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # safety margin before 330s kill
_EPS = 1e-9


def acquisition(mu, sigma, f_best):
    """Return 2000 finite scores; higher => evaluate next.
    
    Uses a Hybrid acquisition function combining Expected Improvement (EI) 
    for exploitation and Lower Confidence Bound (LCB) for exploration.
    The weights are dynamically adjusted based on the relative uncertainty 
    of the posterior distribution.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, _EPS)

    # 1. Compute Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in extreme tails
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # 2. Compute Lower Confidence Bound (LCB) score
    # We want to minimize f. An optimistic bound is mu - beta * sigma.
    # We return a score where higher is better, so we negate the bound:
    # score_lcb = -(mu - beta * sigma) = -mu + beta * sigma
    # We use a fixed beta of 1.0 for this component, but its weight will vary.
    beta = 1.0
    lcb_score = -mu + beta * sigma

    # 3. Dynamic Weighting
    # Estimate the "stage" of optimization.
    # If the mean sigma is large relative to the spread of mu, we are early/uncertain.
    # If mean sigma is small, we are late/certain.
    
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min + _EPS
    
    mean_sigma = np.mean(sigma)
    
    # Relative uncertainty: high if sigma is large compared to the variation in mu
    # This ratio is roughly 1/sqrt(n) scaling if mu converges.
    relative_uncertainty = mean_sigma / mu_range
    
    # We want to explore more when relative_uncertainty is high.
    # Map relative_uncertainty to a weight for LCB (exploration) in [0.0, 1.0].
    # Let's say if relative_uncertainty > 1.0, weight is 1.0 (pure exploration/LCB)
    # if relative_uncertainty < 0.1, weight is 0.0 (pure exploitation/EI)
    # Linear interpolation with a shift.
    # w_lcb = clip(relative_uncertainty * 2.0, 0.0, 1.0)
    
    # A smoother transition might be better.
    # Let's use a sigmoid-like or clipped linear mapping.
    # Early: high sigma/mu_range -> high LCB weight.
    # Late: low sigma/mu_range -> low LCB weight.
    
    # Normalize relative_uncertainty to [0, 1] assuming typical range [0, 2]
    w_lcb = np.clip(relative_uncertainty * 1.5, 0.0, 1.0)
    w_ei = 1.0 - w_lcb
    
    # Ensure weights are positive
    w_lcb = np.maximum(w_lcb, 0.1) # Always keep some exploration
    w_ei = np.maximum(w_ei, 0.1)   # Always keep some exploitation
    
    # Normalize scores to be comparable
    # EI is in units of improvement (positive).
    # LCB score is in units of objective value (can be negative/positive).
    # We should normalize them to [0, 1] or similar scale before combining.
    
    ei_max = np.max(ei) + _EPS
    ei_norm = ei / ei_max
    
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min + _EPS
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # Combine
    scores = w_ei * ei_norm + w_lcb * lcb_norm

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard: if we've spent too long, return a simple robust fallback
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to pure EI which is fast and robust
        z2 = (f_best - mu) / sigma
        z2 = np.clip(z2, -10.0, 10.0)
        scores = (f_best - mu) * norm.cdf(z2) + sigma * norm.pdf(z2)
        scores = np.maximum(scores, 0.0)
        scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
