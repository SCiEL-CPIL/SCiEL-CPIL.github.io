import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma and cost to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Deterministic Worst-Case Improvement (WCI) with cost awareness
    # WCI = max(0, f_best - mu - k*sigma)
    # This is a robust minimax regret approach that penalizes high uncertainty
    # We use a smaller k to reduce over-exploration
    
    k = 0.3  # Exploration constant
    wci = np.maximum(0.0, f_best - mu - k * sigma)
    wci_per_cost = wci / cost_safe
    
    # Also compute a simple LCB (Lower Confidence Bound) for exploitation
    # LCB = mu - k2 * sigma (lower is better for minimization)
    # Score: -LCB / cost = (k2*sigma - mu) / cost
    k2 = 0.5
    lcb_score = (k2 * sigma - mu) / cost_safe
    
    # Normalize both scores
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        s_range = s_max - s_min
        if s_range > 1e-12:
            return (s - s_min) / s_range
        return np.zeros_like(s)
    
    norm_wci = normalize_scores(wci_per_cost)
    norm_lcb = normalize_scores(lcb_score)
    
    # Blend WCI with LCB
    # WCI provides robust exploration, LCB provides exploitation
    w_wci = 0.6
    w_lcb = 0.4
    
    scores = w_wci * norm_wci + w_lcb * norm_lcb
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
    
    return scores
