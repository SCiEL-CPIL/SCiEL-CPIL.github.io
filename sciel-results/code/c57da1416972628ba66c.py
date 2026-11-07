# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Calculate a robust measure of maximum uncertainty
    # Using max is standard, but we can dampen it to avoid excessive exploration
    max_sigma = np.max(sigma)
    
    # Sample an offset from a Gaussian centered at f_best
    # We scale the standard deviation by 0.5 to favor exploitation more strongly
    # in the low-budget regime, while still allowing exploration when uncertainty is high.
    sampling_std = 0.5 * max_sigma
    
    # If uncertainty is extremely low, ensure we don't sample exactly f_best 
    # (which would yield 0 scores for all points with mu > f_best)
    # But since f_best is the minimum, mu >= f_best roughly.
    # If sampling_std is 0, offset = f_best.
    # Score = -(mu - f_best) / cost. This favors low mu (exploitation).
    
    offset = np.random.normal(f_best, sampling_std)
    
    # Compute scores: higher is better.
    # We want to maximize (offset - mu) / cost.
    # This is equivalent to minimizing (mu - offset) / cost.
    # If offset < mu, score is negative. If offset > mu, score is positive.
    # Since offset is centered at f_best (the min), and mu >= f_best typically,
    # most scores will be negative unless we sample an offset above some mu.
    # Actually, we want to find points where the true function value is low.
    # TS interpretation: We sample a possible true value for each point? 
    # No, the directive says: "sample a single scalar offset... score as -(mu - offset) / cost".
    # This is a specific heuristic.
    
    scores = -(mu - offset) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
