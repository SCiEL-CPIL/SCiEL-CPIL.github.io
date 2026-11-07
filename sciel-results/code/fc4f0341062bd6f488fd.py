"""Rewritten program for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track wall-clock deadline
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to leave before hard stop

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI) and
    Lower Confidence Bound (LCB) with adaptive exploration parameter.
    
    This approach balances exploitation (via EI) and exploration (via LCB).
    The exploration parameter beta is adapted based on the current uncertainty
    relative to the best observed value, encouraging more exploration when
    uncertainty is high and less when the model is confident.
    
    Returns scores where higher values indicate better candidates.
    """
    # Check time to ensure we don't exceed limits if called in a tight loop or slow environment
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    # If we are extremely close to the deadline, use a very fast heuristic
    if remaining_time < _DEADLINE_MARGIN:
        # Heuristic: Lower mu is better. Some exploration via sigma.
        scores = -mu - 0.5 * sigma
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores

    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI) for minimization
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Calculate Lower Confidence Bound (LCB) for minimization
    # LCB = mu - beta * sigma
    # We want to maximize the score, so we use -LCB = -mu + beta * sigma
    # Adaptive beta: larger when uncertainty is high relative to the range of f_best
    # Use a simple adaptive scheme: beta = 0.5 + 0.5 * (sigma_safe / (sigma_safe + abs(f_best) + eps))
    # This makes beta larger when sigma is large relative to the scale of f_best
    beta = 0.5 + 0.5 * (sigma_safe / (sigma_safe + np.abs(f_best) + eps))
    lcb_score = -mu + beta * sigma_safe
    
    # Combine EI and LCB scores
    # Normalize both to have similar scales
    # Use a weighted combination: 0.5 * EI + 0.5 * LCB_score
    # But we need to ensure they are on comparable scales
    # EI can be negative, LCB_score can be anything
    # Let's use a different approach: use EI as the primary signal and add a small exploration bonus
    # Or use a max of the two, or a sum with normalization
    
    # Better approach: Use a single unified acquisition that combines both
    # Let's try: score = EI + gamma * (sigma_safe / (sigma_safe + eps))
    # where gamma is a small exploration weight
    # Actually, let's use the standard approach: use EI with a small epsilon to avoid division by zero
    # and add a small amount of exploration via sigma
    
    # Let's try a different structural change: use Upper Confidence Bound for maximization
    # which is equivalent to LCB for minimization, but with a different formulation
    # UCB for minimization: -mu + beta * sigma
    # Let's use a fixed beta that works well for 2D problems
    # For 2D, beta around 1.0-2.0 often works well
    
    # Let's use a combination: 0.7 * EI + 0.3 * (-mu + 1.5 * sigma)
    # But we need to normalize EI to be comparable to the LCB term
    
    # Alternative: Use the raw EI but with a modified sigma that includes a floor
    # This is what the original does, but let's try a different acquisition entirely
    
    # Let's try Thompson Sampling approximation:
    # Sample from the posterior: x ~ N(mu, sigma^2)
    # But we can't sample per candidate efficiently for 2000 points
    # Instead, use the expected value of the minimum, which is related to EI
    
    # Let's try a simpler but different approach: Use the probability of improvement
    # P(f(x) < f_best) = Phi(z)
    # This is a valid acquisition function that favors points likely to beat f_best
    # But it doesn't consider the magnitude of improvement
    
    # Let's use a hybrid: 0.5 * EI + 0.5 * Phi(z) * (f_best - mu)
    # This combines the expected improvement with the probability of improvement
    
    # Actually, let's try the most standard and robust approach:
    # Use EI but with a small epsilon added to sigma to avoid numerical issues
    # and ensure the scores are well-behaved
    
    # Let's try a different structural change: use the "Expected Regret" which is
    # E[max(0, f(x) - f_best)] = -EI (for minimization)
    # So minimizing expected regret is equivalent to maximizing EI
    
    # Let's try using a quantile-based approach:
    # Use the 10th percentile of the posterior as a robust estimate
    # q10 = mu - 1.28 * sigma
    # This is more robust to outliers than the mean
    
    # Let's combine multiple signals:
    # 1. EI (exploitation)
    # 2. LCB with beta=1.0 (exploration)
    # 3. Probability of improvement (confidence)
    
    # Normalize each to have zero mean and unit variance, then combine
    ei_mean = np.nanmean(ei)
    ei_std = np.nanstd(ei) + eps
    ei_norm = (ei - ei_mean) / ei_std
    
    lcb_raw = -mu + 1.0 * sigma_safe
    lcb_mean = np.nanmean(lcb_raw)
    lcb_std = np.nanstd(lcb_raw) + eps
    lcb_norm = (lcb_raw - lcb_mean) / lcb_std
    
    p_imp = Phi_z
    p_imp_mean = np.nanmean(p_imp)
    p_imp_std = np.nanstd(p_imp) + eps
    p_imp_norm = (p_imp - p_imp_mean) / p_imp_std
    
    # Combine with weights: 0.5 * EI + 0.3 * LCB + 0.2 * P(imp)
    scores = 0.5 * ei_norm + 0.3 * lcb_norm + 0.2 * p_imp_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
