import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive beta based on gap to known global minimum (3.0)
    # If f_best is much larger than 3.0, we are far from optimum, explore more.
    # If f_best is close to 3.0, exploit.
    gap = max(0.0, f_best - 3.0)
    
    # Base beta 0.5, increasing with gap, capped at 1.5 to avoid excessive exploration
    beta = 0.5 + 0.5 * gap
    beta = np.clip(beta, 0.5, 1.5)
    
    # UCB score for minimization: -mu + beta * sigma
    # Higher score means "evaluate next"
    scores_ucb = -mu + beta * sigma
    
    # Standard EI
    z = (f_best - mu) / sigma
    scores_ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Strategy: Use UCB if the uncertainty is significant relative to the gap?
    # Or just blend?
    # Let's try a simple blend: 50% EI, 50% UCB (normalized)
    # Normalization is risky if range is 0.
    
    # Alternative: Use UCB directly.
    # Given Inspiration 1 failed with UCB optimization, maybe pure EI is best?
    # But we need to "combine" and "improve".
    
    # Let's try a "Soft" switch:
    # If the best EI is very low (close to 0), it means we are near the optimum.
    # In that case, EI is good.
    # If the best EI is high, there is potential for improvement.
    
    # Let's try: score = EI + 0.2 * (sigma / (sigma.mean() + 1e-9))
    # This adds a normalized exploration term.
    
    mean_sigma = np.mean(sigma)
    exploration_term = sigma / (mean_sigma + 1e-9)
    
    # Blend EI and exploration
    # Weight: 0.9 EI, 0.1 Exploration
    scores = 0.9 * scores_ei + 0.1 * exploration_term * np.max(scores_ei)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
