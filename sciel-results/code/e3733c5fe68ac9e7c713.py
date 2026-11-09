"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for 2000 candidates.
    Uses a log-transformed Lower Confidence Bound (LCB) with a logarithmic cost penalty.
    score = log(LCB) - 0.5 * log(cost)
    This models optimization as multiplicative utility maximization, preventing
    aggressive bias toward low-cost regions in early search.
    """
    # Ensure finite inputs; replace non-finite with safe defaults
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.clip(sigma, 1e-9, None)
    
    # Cost should be positive
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)

    # Lower Confidence Bound for minimization: LCB = mu - kappa * sigma
    # We want to maximize the acquisition, so lower LCB is better
    kappa = 2.0
    lcb = mu - kappa * sigma

    # For log transform, LCB must be positive. 
    # If LCB <= 0, we add a small offset to make it positive.
    # The offset should be relative to the scale of the problem.
    # f_best is typically negative for Styblinski-Tang (min ~ -78).
    # Let's use a shift that ensures positivity while preserving order.
    # A safe shift is max(0, -f_best + 1) + 1, or simply a constant if f_best is very negative.
    # Actually, since we want to maximize log(LCB), and LCB can be negative,
    # we can shift by a constant C such that LCB + C > 0.
    # C = max(0, -min(lcb) + 1) would work but depends on batch min.
    # Simpler: shift by a large constant? No, that distorts.
    # Better: use the fact that we are maximizing. 
    # If LCB is negative, log is undefined.
    # Alternative: use log1p or a shifted log.
    # Let's use: log(lcb - min_lcb + epsilon) ? No, min_lcb changes per batch.
    
    # Let's stick to the directive's formula: log(mu - 2.0 * sigma) - 0.5 * log(cost)
    # But we must handle negative arguments to log.
    # We can add a constant offset to the LCB before taking log.
    # The offset must be large enough to make all LCB values positive.
    # Since f_best is the best observed, and mu is close to f_best, 
    # and sigma is small, LCB is likely near f_best.
    # If f_best is negative, LCB is negative.
    # Let's shift by -f_best + 1. This makes the best point have value ~1, and others lower.
    # Wait, if f_best is -78, shift is 79. LCB is ~-78. LCB + 79 = 1.
    # If LCB is -100, LCB + 79 = -21. Still negative.
    
    # Robust approach:
    # 1. Calculate LCB.
    # 2. Find the minimum LCB in the batch.
    # 3. Shift all LCBs by -min_lcb + epsilon.
    # This preserves relative ordering and ensures positivity.
    
    min_lcb = np.min(lcb)
    shift = -min_lcb + 1.0  # Ensure at least 1.0
    lcb_shifted = lcb + shift
    
    # Now take log
    log_lcb = np.log(lcb_shifted)
    log_cost = 0.5 * np.log(cost)
    
    score = log_lcb - log_cost

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
