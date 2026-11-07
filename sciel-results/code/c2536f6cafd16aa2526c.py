"""Acquisition function using Lower Confidence Bound with adaptive beta."""
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # Kill after 330s wall-clock, use 320s as safety margin


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition with adaptive beta.
    
    beta scales inversely with f_best: when f_best is large (early),
    we explore more; when f_best is small (late), we exploit more.
    
    score = -mu - beta * sigma
    Higher score means more likely to be the minimum.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive beta: larger when f_best is large (early), smaller when f_best is small (late)
    # beta = 1 + 10 / max(f_best, 0.01)
    # This gives beta ~ 1001 when f_best ~ 0.01, and beta ~ 2 when f_best ~ 10
    # But we want the opposite: more exploration early (large f_best), less late (small f_best)
    # So: beta = 1 + 10 / (1 + f_best) gives beta ~ 11 when f_best=0, beta ~ 2 when f_best=9
    
    # Actually, for LCB in minimization: we want to explore when uncertain
    # score = mu + beta * sigma would be for maximization
    # For minimization: score = -mu - beta * sigma, so higher score = lower mu + more exploration
    
    # Let's use: score = -(mu + beta * sigma) where beta adapts
    # When f_best is large, we have more room to explore, so larger beta
    # When f_best is small, we should exploit, so smaller beta
    
    # beta = 1 + 5 * (1 / (1 + f_best))
    # f_best = 100: beta ≈ 1.05 (mostly exploitation)
    # f_best = 1: beta ≈ 3.5 (balanced)
    # f_best = 0.01: beta ≈ 6 (more exploration)
    
    # Wait, this is backwards. Early on f_best is large, we want MORE exploration.
    # So beta should be LARGER when f_best is LARGE.
    
    # beta = 1 + 5 * (f_best / (1 + f_best))
    # f_best = 100: beta ≈ 6 (more exploration)
    # f_best = 1: beta ≈ 3.5 (balanced)
    # f_best = 0.01: beta ≈ 1.05 (mostly exploitation)
    
    # But this might be too aggressive. Let's use a smoother transition.
    # beta = 1 + 3 * (f_best / (1 + f_best))
    
    f_best_clipped = max(f_best, 1e-6)
    beta = 1.0 + 3.0 * (f_best_clipped / (1.0 + f_best_clipped))
    
    # LCB score for minimization: we want to maximize -(mu + beta * sigma)
    # Equivalent to minimizing mu + beta * sigma
    # So the acquisition score (higher is better) is:
    score = -(mu + beta * sigma)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Small regularization to avoid exact ties
    score += 1e-12 * np.random.randn(len(score))
    
    return score
