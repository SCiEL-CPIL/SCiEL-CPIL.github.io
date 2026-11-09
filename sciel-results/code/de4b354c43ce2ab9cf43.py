"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE_SECONDS = 325.0

def _expected_improvement(mu, sigma, f_best):
    """Standard Gaussian Expected Improvement."""
    sigma_safe = np.maximum(sigma, 1e-10)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei

def _ucb(mu, sigma, beta):
    """Upper Confidence Bound (for minimization: lower is better)."""
    sigma_safe = np.maximum(sigma, 1e-10)
    return -mu + beta * sigma_safe

def _probability_improvement(mu, sigma, f_best):
    """Probability of Improvement."""
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)

def _log_ei(mu, sigma, f_best):
    """Log-Expected Improvement for numerical stability."""
    sigma_safe = np.maximum(sigma, 1e-10)
    improvement = f_best - mu
    z = improvement / sigma_safe
    # Use log-sum-exp style for numerical stability
    # log(EI) = log(improvement * Phi(z) + sigma * phi(z))
    # Handle cases where improvement <= 0
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Clip to avoid log of negative or zero
    ei_safe = np.maximum(ei, 1e-15)
    return np.log(ei_safe)

def acquisition(mu, sigma, f_best):
    """
    Ensemble acquisition: compute multiple acquisition functions and
    return the pointwise maximum. This lets diverse strategies compete
    at every candidate location.
    
    Structural design:
    - EI: exploitation-focused
    - UCB with multiple beta values: exploration-exploitation tradeoff
    - PI: pure probability-based
    - Log-EI: stable variant
    
    A progressive schedule shifts weight from exploration to exploitation
    as f_best decreases (proxy for progress in the 30-eval budget).
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return -mu

    sigma_safe = np.maximum(sigma, 1e-10)
    n = len(mu)

    # Progress proxy: f_best relative to Beale function range.
    # Beale on [-4.5,4.5]^2: typical initial values ~100-1000, min=0.
    # Map f_best to a progress in [0, 1] where 0=early, 1=late.
    # Use a soft mapping: progress = 1 - exp(-f_best / scale)
    # At f_best=1000, scale=200: progress ~ 1-exp(-5) ~ 0.99 (late)
    # At f_best=100, scale=200: progress ~ 1-exp(-0.5) ~ 0.39 (mid)
    # At f_best=10, scale=200: progress ~ 1-exp(-0.05) ~ 0.05 (early)
    # Actually we want early=high exploration, so let's flip:
    # exploration_weight = 1 - progress
    # But f_best starts high (~500-2000) and goes to 0.
    # Let's use: exploration_weight = min(1.0, f_best / 200.0)
    # When f_best > 200, full exploration weight.
    # When f_best < 200, decreasing exploration.
    exploration_weight = min(1.0, max(0.0, f_best / 200.0))

    # Compute each acquisition
    ei_scores = _expected_improvement(mu, sigma_safe, f_best)
    pi_scores = _probability_improvement(mu, sigma_safe, f_best)
    log_ei_scores = _log_ei(mu, sigma_safe, f_best)

    # UCB with multiple betas, scaled by exploration weight
    # Early: high beta (more exploration), Late: low beta (more exploitation)
    beta_early = 2.0 * exploration_weight + 0.1
    beta_mid = 1.0 * exploration_weight + 0.05
    beta_late = 0.5 * exploration_weight + 0.02

    ucb_early = _ucb(mu, sigma_safe, beta_early)
    ucb_mid = _ucb(mu, sigma_safe, beta_mid)
    ucb_late = _ucb(mu, sigma_safe, beta_late)

    # Normalize scores to comparable scales for ensemble selection.
    # EI is in objective units (can be large early).
    # UCB is in objective units (negative mu + beta*sigma).
    # PI is in [0,1].
    # Log-EI is unbounded but typically smaller.
    #
    # Strategy: return the pointwise maximum of all normalized scores.
    # To make them comparable, we can:
    # 1. Use raw EI (dominant in exploitation)
    # 2. Use raw UCB (dominant in exploration)
    # 3. Blend: take max of (EI, UCB_early, UCB_mid, UCB_late)
    #    This naturally picks exploration when EI is low but uncertainty is high.
    #
    # A cleaner structural approach: compute a weighted sum where weights
    # shift with progress, then return that. But pointwise max is more robust
    # to scale mismatches.
    #
    # Let's use: score = max(EI, UCB_early, UCB_mid, UCB_late)
    # This ensures we always have at least the EI signal, and UCB can override
    # when uncertainty is high enough.

    scores = np.maximum(ei_scores, ucb_early)
    scores = np.maximum(scores, ucb_mid)
    scores = np.maximum(scores, ucb_late)

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    # Replace -inf with a very negative finite value
    scores = np.where(np.isfinite(scores), scores, -1e10)

    return scores

# EVOLVE-BLOCK-END
