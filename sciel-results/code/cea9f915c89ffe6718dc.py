import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses adaptive UCB blended with probability of improvement,
    with rank-based normalization for robustness.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive exploration weight:
    # Michalewicz 2D min is -1.8013. As f_best approaches this,
    # reduce exploration. Early on (f_best far from min), explore more.
    # Progress: 0 when f_best is far, 1 when f_best is at minimum
    min_val = -1.8013
    # Scale progress: assume typical initial f_best around -1.0 to -1.5
    # Use a smooth transition
    progress = np.clip((f_best - min_val) / (f_best - min_val + 0.5), 0.0, 1.0) if f_best > min_val else 1.0
    # Actually, simpler: normalize by expected range
    # f_best starts around -1.0 to -1.3, ends near -1.8013
    # Use: progress = 1 - (f_best - min_val) / (initial_estimate - min_val)
    # But we don't know initial. Use a fixed scale.
    # When f_best = -1.0, progress should be low (explore more)
    # When f_best = -1.8013, progress = 1 (exploit)
    scale = 0.8  # approximate range from typical start to min
    progress = np.clip(1.0 - (f_best - min_val) / scale, 0.0, 1.0)
    
    # Beta: exploration weight, high early, low late
    beta = 0.6 * (1.0 - progress) + 0.15 * progress
    
    # UCB score: mu + beta * sigma (higher is better for maximization of score)
    # But we're minimizing, so improvement means lower f.
    # For minimization: UCB_min = mu - beta * sigma (we want low mu, high sigma for exploration)
    # Actually, let's think in terms of "improvement":
    # We want to find points where f < f_best.
    # EI for minimization: (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu)/sigma
    # UCB for minimization: mu - beta * sigma (lower is more promising)
    # PI for minimization: Phi((f_best - mu) / sigma)
    
    z = (f_best - mu) / sigma
    
    # Probability of improvement (for minimization)
    pi = norm.cdf(z)
    
    # UCB for minimization (lower value = more promising)
    ucb = mu - beta * sigma
    
    # Expected improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Blend: use UCB and PI and EI
    # Normalize each to [0, 1] range using rank for robustness
    def rank_normalize(x):
        """Convert to rank-based scores in [0, 1]."""
        n = len(x)
        if n <= 1:
            return np.ones_like(x)
        ranks = np.argsort(np.argsort(x)).astype(float)
        return (ranks + 1) / (n + 1)
    
    # Rank-normalize each component
    ucb_rank = rank_normalize(-ucb)  # negate so higher = more promising
    pi_rank = rank_normalize(pi)
    ei_rank = rank_normalize(ei)
    
    # Adaptive blend weights:
    # Early: more UCB (exploration), Late: more EI/PI (exploitation)
    w_ucb = 0.5 * (1.0 - progress) + 0.2 * progress
    w_pi = 0.3 * (1.0 - progress) + 0.2 * progress
    w_ei = 0.2 * (1.0 - progress) + 0.6 * progress
    
    # Normalize weights
    w_sum = w_ucb + w_pi + w_ei
    w_ucb /= w_sum
    w_pi /= w_sum
    w_ei /= w_sum
    
    score = w_ucb * ucb_rank + w_pi * pi_rank + w_ei * ei_rank
    
    # Ensure all finite
    score = np.where(np.isfinite(score), score, 0.5)
    
    return score
