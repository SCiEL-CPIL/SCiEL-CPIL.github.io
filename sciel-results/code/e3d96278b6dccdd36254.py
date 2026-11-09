import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Potential improvement magnitude (clipped at 0)
    improvement = np.maximum(0, f_best - mu)
    
    # Weighted exploitation: POI * Improvement
    # This term is high when we are confident the point is better, and the improvement is large.
    exploit_term = poi * improvement
    
    # Exploration term: Sigma
    # This term is high when uncertainty is high.
    explore_term = sigma_safe
    
    # Combine
    # We can tune the relative weight of exploration vs exploitation.
    # Currently 1:1.
    # If exploration is too high, we might wander.
    # If exploitation is too high, we might get stuck in local optima.
    # Given Levy 2D is multimodal, some exploration is needed.
    # But regret is dominated by the best point found.
    # Let's stick to 1:1 for now, but ensure the scales are similar.
    # 'improvement' and 'sigma' are in objective units.
    
    combined = exploit_term + explore_term
    
    # Normalize by cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = combined / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
