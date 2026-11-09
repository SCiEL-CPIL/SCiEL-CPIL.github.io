import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is not zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    # E[max(0, f_best - f(x))]
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize f(x). LCB is mu - beta * sigma.
    # To make it a "higher is better" score, we use -LCB = beta * sigma - mu.
    beta = 1.5
    lcb_score = beta * sigma_safe - mu
    
    # Blend EI and LCB scores
    # Weights: 0.6 for EI, 0.4 for LCB
    w_ei = 0.6
    w_lcb = 0.4
    
    raw_score = w_ei * ei + w_lcb * lcb_score
    
    # Cost-aware: divide by cost
    # Cost is passed as an array of shape (2000,)
    # Ensure cost is not zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = raw_score / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
