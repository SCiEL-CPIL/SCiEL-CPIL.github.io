"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time for deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # 10 seconds before the 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a hybrid approach combining:
    1. Expected Improvement (EI) for exploitation
    2. Knowledge Gradient-like exploration via uncertainty
    3. Adaptive weighting based on convergence progress
    
    The Weierstrass function is highly rugged with many local minima.
    Early on, we need strong exploration to find good basins.
    Later, we need precise exploitation to converge to the minimum.
    """
    # Deadline guard: if we're close to the limit, return a safe score
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a simple deterministic score favoring low mu
        return -mu.astype(np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best + 1e6)
    
    # Compute Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute a Knowledge Gradient-like score
    # The idea: the value of information from evaluating at x is related to
    # how much it could improve our belief about the minimum.
    # Approximate: favor points where sigma is high AND mu is competitive.
    
    # Normalize mu to get a sense of how good each point is
    # Weierstrass 1D on [-0.5, 0.5] has minimum 0, and typical values are O(1)
    
    # Compute a "promise" score: how much lower could this point be?
    # Lower bound estimate: mu - k * sigma (optimistic bound)
    # We want points where this lower bound is low (good) and uncertainty is high (informative)
    
    # Adaptive beta based on progress:
    # f_best starts high (maybe ~1-2) and decreases toward 0
    # When f_best is large, explore more; when small, exploit more
    
    # Use a sigmoid-like transition based on f_best
    # f_best near 0 -> exploitation (low beta)
    # f_best large -> exploration (high beta)
    beta_base = 2.0
    # Scale: when f_best = 0, beta = beta_base * 0.1
    #        when f_best = 1, beta = beta_base * 0.5
    #        when f_best = 2, beta = beta_base * 0.9
    #        when f_best > 3, beta = beta_base
    progress = 1.0 / (1.0 + np.exp(2.0 * (f_best - 0.5)))
    # progress is near 0 when f_best >> 0.5, near 1 when f_best << 0.5
    beta = beta_base * (0.1 + 0.9 * (1.0 - progress))
    
    # UCB score for minimization: -mu + beta * sigma
    ucb = -mu + beta * sigma
    
    # Also compute a "lower confidence bound" style score for exploration
    # LCB = mu + beta * sigma (for minimization, we want low LCB)
    # But we want to MAXIMIZE the acquisition, so:
    # Exploration score = -mu + beta * sigma (same as UCB above)
    
    # Now blend EI and UCB with adaptive weights
    # When f_best is large (early), weight UCB more
    # When f_best is small (late), weight EI more
    
    # Weight for EI: increases as we converge
    # Weight for UCB: decreases as we converge
    w_ei = 0.3 + 0.7 * progress
    w_ucb = 1.0 - w_ei
    
    # Normalize both scores to zero mean, unit variance for stable blending
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-9
    
    ei_norm = (ei - ei_mean) / ei_std
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Weighted combination
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Add a small diversity bonus: favor points that are in regions with
    # high uncertainty relative to the local neighborhood
    # This helps escape local optima in the rugged Weierstrass landscape
    # Simple approximation: use the raw sigma as a diversity signal
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-9
    diversity = (sigma - sigma_mean) / sigma_std
    
    # Small weight for diversity to avoid completely ignoring uncertain regions
    score += 0.1 * diversity
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
