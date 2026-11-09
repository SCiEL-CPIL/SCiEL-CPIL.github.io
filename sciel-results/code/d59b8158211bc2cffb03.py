"""Acquisition function for Bayesian optimization of weierstrass 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guarding
_START_TIME = time.time()
# Safety margin in seconds (300s limit, leave 10s buffer)
_DEADLINE_MARGIN = 290.0

def acquisition(mu, sigma, f_best):
    """
    Global Virtual Population Acquisition Function.
    
    Uses a Shapley-like marginal contribution score based on the probability
    that a candidate is the minimum of the set AND improves upon f_best.
    Assumes independence between candidates for the pairwise comparison
    (model-free approximation of the joint posterior structure).
    
    Args:
        mu: array of shape (2000,) with posterior means.
        sigma: array of shape (2000,) with posterior standard deviations.
        f_best: scalar, the lowest observed value so far.
        
    Returns:
        array of shape (2000,) with acquisition scores (higher is better).
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_MARGIN:
        # Fallback to simple EI if time is critical
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -8.0, 8.0)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    n = len(mu)
    
    # 1. Probability of improving on f_best (Point-wise)
    z_best = (f_best - mu) / sigma
    z_best_clipped = np.clip(z_best, -8.0, 8.0)
    p_improve = norm.cdf(z_best_clipped)
    
    # 2. Pairwise dominance probabilities
    # P(f_i < f_j) = Phi((mu_j - mu_i) / sqrt(sigma_i^2 + sigma_j^2))
    
    # Vectorized calculation
    # mu_j - mu_i : (N, 1) - (1, N) -> (N, N)
    diff_mu = mu[None, :] - mu[:, None]
    
    # sigma_i^2 + sigma_j^2 : (N, 1) + (1, N) -> (N, N)
    var_sum = sigma[:, None]**2 + sigma[None, :]**2
    std_sum = np.sqrt(var_sum)
    
    # Avoid division by zero (though sigma is max 1e-9, so std_sum > 0)
    z_pair = diff_mu / std_sum
    
    # Clip for numerical stability
    z_pair_clipped = np.clip(z_pair, -8.0, 8.0)
    
    # Probability that i beats j
    p_beat = norm.cdf(z_pair_clipped)
    
    # 3. Marginal contribution to being the minimum of the set
    # We want the probability that i is the minimum among all j != i.
    # Under independence assumption: P(i is min) = prod_{j != i} P(i beats j)
    # Log space: log P(i is min) = sum_{j != i} log(P(i beats j))
    
    # Mask the diagonal (i vs i) which is 0.5 (Phi(0))
    # We set log(0.5) to 0 to exclude self-comparison from the product
    log_p_beat = np.log(p_beat + 1e-15)
    
    # Zero out the diagonal
    np.fill_diagonal(log_p_beat, 0.0)
    
    # Sum over columns (for each i, sum over j)
    # log_p_beat[i, j] is log P(f_i < f_j)
    # We want sum_j log P(f_i < f_j)
    log_p_min = np.sum(log_p_beat, axis=1)
    
    # Convert back to probability scale, but normalize to avoid underflow
    # The absolute value is very small, so we use the relative ranking.
    # Score_i = p_improve[i] * exp(log_p_min[i] - max(log_p_min))
    # This normalizes the "min-ness" factor to [0, 1] range roughly.
    
    max_log_p_min = np.max(log_p_min)
    # If all log_p_min are -inf (shouldn't happen with 1e-15), handle it
    if not np.isfinite(max_log_p_min):
        min_factor = np.ones(n)
    else:
        # exp(log_p_min - max) gives relative weight, max is 1.0
        min_factor = np.exp(log_p_min - max_log_p_min)
        
    # Combine: We want points that improve on best AND are likely to be the unique minimum
    # This penalizes clusters of similar high-EI points.
    scores = p_improve * min_factor
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
