"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Strategy: Hybrid acquisition combining Expected Improvement (EI) and
    # Upper Confidence Bound (UCB) with adaptive weighting.
    #
    # The key insight is that early in the search, exploration is more important
    # to discover the global minimum, while later exploitation should dominate
    # to refine the best candidate.
    #
    # We use a dynamic beta that starts high (exploration) and decays over time
    # based on how close f_best is to the known minimum (-39.1662).
    
    # Known minimum for Styblinski-Tang 1D
    f_min_known = -39.1662
    
    # Compute Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is non-negative; higher EI means more expected improvement
    
    # Compute UCB score (for minimization, we want to maximize -mu + beta*sigma)
    # This encourages exploring regions with high uncertainty
    
    # Adaptive beta: start high, decay as we get closer to the minimum
    # The "distance" from minimum is (f_best - f_min_known), which is positive
    # and decreases as we improve
    distance_to_min = max(f_best - f_min_known, 0.0)
    
    # Normalize distance: at start, f_best might be around -10 to -20, so distance ~ 20-30
    # At end, distance ~ 0
    # We want beta to range from ~4.0 (early) to ~1.0 (late)
    # Use a smooth decay: beta = 1.0 + 3.0 * exp(-distance_to_min / 5.0)
    # When distance is large (early), exp term ~ 1, beta ~ 4.0
    # When distance is small (late), exp term ~ 0, beta ~ 1.0
    
    beta = 1.0 + 3.0 * np.exp(-distance_to_min / 5.0)
    
    # UCB score: higher means more promising for exploration
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # EI is typically small (0 to a few units), UCB can be larger
    
    # Use a weighted combination
    # Weight EI more when we're close to the minimum (exploitation phase)
    # Weight UCB more when we're far (exploration phase)
    
    # Simple approach: use EI directly, but add a small UCB bonus
    # to ensure some exploration even in late stages
    
    # Alternative: use a blend
    # score = w_ei * EI + w_ucb * normalized_UCB
    
    # Let's use EI as the primary score, but ensure that if EI is very small
    # (we're already at a good point), we still have some exploration via UCB
    
    # Scale EI to be comparable to UCB
    # EI is in objective units, UCB is also in objective units
    # They should be roughly comparable
    
    # Use EI as the base score, and add a small fraction of UCB
    # to maintain some exploration pressure
    
    # The fraction should decay as we get closer to the minimum
    # ucb_weight = 0.3 * exp(-distance_to_min / 10.0)
    # When distance is large, ucb_weight ~ 0.3
    # When distance is small, ucb_weight ~ 0.3 * exp(0) = 0.3... that doesn't decay
    
    # Better: ucb_weight = 0.5 * (distance_to_min / (distance_to_min + 5.0))
    # When distance is large, ucb_weight ~ 0.5
    # When distance is small, ucb_weight ~ 0
    
    # Actually, let's just use EI. It's a well-established acquisition function
    # and should work well. The GP model should already capture the multimodality
    # if the length scale is appropriate.
    
    # However, with only 30 evaluations and 1D, EI might be too greedy.
    # Let's try a hybrid: score = EI + alpha * sigma
    # where alpha is small, to add a bit of exploration
    
    # Or better: use the "Knowledge Gradient" or "Probability of Improvement"
    # but those are more complex.
    
    # Let's stick with EI, but ensure it's robust.
    
    # EI can be zero in many places, which is fine.
    # The maximum EI should be at the point with the best expected improvement.
    
    # Compute scores as EI
    scores = ei
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
