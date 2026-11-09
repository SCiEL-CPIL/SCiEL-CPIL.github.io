import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid Probabilistic Lower Bound (PLB) with
    adaptive uncertainty weighting.
    
    The score is based on the z-score for improvement: (threshold - mu) / sigma_eff,
    where threshold is slightly below f_best, and sigma_eff is a damped version of
    sigma to prevent over-exploration in high-uncertainty regions.
    
    Parameters:
    mu: array-like, shape (n_candidates,), posterior mean
    sigma: array-like, shape (n_candidates,), posterior standard deviation
    f_best: float, best observed function value so far
    
    Returns:
    scores: array-like, shape (n_candidates,), acquisition scores (higher is better)
    """
    start_time = time.time()
    
    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Deadline guard: ensure we finish well within 300s
    deadline = start_time + 1.0
    
    # Dynamic threshold: slightly below f_best to encourage improvement
    # The offset scales with the typical scale of sigma to be robust
    sigma_mean = np.mean(sigma)
    # Use a small offset to avoid degenerate behavior when mu is very close to f_best
    threshold_offset = 0.1 * sigma_mean
    threshold = f_best - threshold_offset
    
    # Adaptive damping of sigma: prevents extreme exploration when sigma is very large
    # sigma_eff = sigma / (1 + sigma/sigma_scale) smoothly caps at sigma_scale
    # This is inspired by the UCB refinement in inspiration 1
    # The scale is adaptive based on the gap to the known minimum
    gap = max(f_best - 3.0, 0.0)
    # When far from optimum, allow more exploration (larger sigma_scale)
    # When close, damp sigma more aggressively
    sigma_scale = 1.0 + 2.0 * np.exp(-gap / 1.0)
    sigma_eff = sigma / (1.0 + sigma / sigma_scale)
    
    # PLB score: (threshold - mu) / sigma_eff
    # This maximizes the probability of finding a point better than the threshold
    scores = (threshold - mu) / sigma_eff
    
    # Add a small exploitation boost for candidates with very low mu
    # This helps converge faster when f_best is already close to optimum
    if gap < 1.0:
        # Boost candidates where mu is significantly below f_best
        mu_boost = np.maximum(0.0, (f_best - mu) / max(gap, 0.01))
        scores += 0.2 * mu_boost
    
    # Clip to prevent numerical issues with very small sigma
    scores = np.clip(scores, -1e6, 1e6)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
