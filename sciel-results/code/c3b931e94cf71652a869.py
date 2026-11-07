import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add an exploration bonus (soft UCB component)
    # This helps escape local optima in multimodal functions by rewarding uncertainty
    # even when the mean prediction is not better than f_best.
    exploration_bonus = 0.5 * sigma_safe
    
    # Total acquisition score
    score = ei + exploration_bonus
    
    # Cost-aware scaling
    # Divide by cost to prioritize cheaper evaluations
    # Use safe division to avoid inf if cost is 0 (though cost >= 1)
    cost_safe = np.maximum(cost, 1e-9)
    
    final_score = score / cost_safe
    
    # Ensure finite scores
    return np.nan_to_num(final_score, nan=0.0, posinf=1e10, neginf=-1e10)
