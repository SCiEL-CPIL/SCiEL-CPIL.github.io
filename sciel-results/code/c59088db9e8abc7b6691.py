# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far (minimum)
    cost : array of shape (2000,)
        Evaluation costs for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB (Upper Confidence Bound) - for minimization, we want lower values
    # So use a "lower confidence bound" style: mu - kappa * sigma
    # But for acquisition, higher is better, so we want to explore where sigma is high
    # Let's use a hybrid: combine EI with an exploration bonus
    # Exploration bonus: sigma / cost (prefer high uncertainty with low cost)
    
    cost_safe = np.maximum(cost, 1e-6)
    
    # Normalize cost to [0, 1] range for better scaling
    # cost is in [1, 2], so subtract 1 and divide by 1
    cost_norm = np.clip((cost_safe - 1.0), 0.0, 1.0)
    
    # Adaptive exploration: use a kappa that depends on the relative uncertainty
    # When sigma is large relative to (f_best - mu), exploration should be stronger
    # Compute a measure of how "uncertain" each point is relative to improvement
    improvement = np.maximum(f_best - mu, 1e-8)
    uncertainty_ratio = sigma_safe / improvement
    
    # kappa: higher when uncertainty is high relative to potential improvement
    # Clip to avoid extreme values
    kappa = 0.5 * np.clip(uncertainty_ratio, 0.0, 5.0)
    
    # UCB-style score: (f_best - mu - kappa * sigma) / cost
    # This is higher when mu is low (good) and sigma is low (certain), 
    # but we want to explore, so actually for minimization:
    # We want to pick points with low mu (likely good) and high sigma (uncertain)
    # So: score = (f_best - mu) / cost + kappa * sigma / cost
    # Or: score = ei / cost + kappa * sigma / cost
    
    # Let's use: score = (ei + kappa * sigma) / cost^beta
    # where beta is an adaptive cost exponent
    
    # Adaptive beta: when cost varies a lot, use a softer penalty
    cost_var = np.var(cost_safe)
    # If cost variance is high, use beta closer to 0.5; if low, use beta closer to 1.0
    # cost_var is in [0, 0.25] roughly for cost in [1,2]
    beta = 0.5 + 0.5 * np.clip(1.0 - cost_var * 4.0, 0.0, 1.0)
    
    # Compute scores
    exploration_bonus = kappa * sigma_safe
    combined = ei + exploration_bonus
    scores = combined / (cost_safe ** beta)
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
