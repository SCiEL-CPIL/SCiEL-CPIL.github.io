import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Adaptive Lower Confidence Bound (LCB) acquisition function.
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    deadline = 295.0  # safety margin before 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost = np.maximum(cost, 1e-8)
    
    # Normalize sigma for stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive LCB: LCB = mu - beta * sigma
    # For minimization, lower LCB is better, so we want to MAXIMIZE -LCB = -mu + beta*sigma
    # Or equivalently, score = beta * sigma - mu (higher is better)
    
    # Normalize f_best to estimate progress
    # Ackley 2D max is approx 29.
    # Use a safe upper bound.
    f_scale = 25.0
    f_ratio = np.clip(f_best / f_scale, 0.0, 1.0)
    
    # Beta: higher when f_best is high (early, more exploration), 
    # lower when f_best is low (late, more exploitation)
    beta_min = 0.5
    beta_max = 3.0
    beta = beta_min + (beta_max - beta_min) * f_ratio
    
    # Cost-aware LCB: We want to maximize expected improvement per unit cost.
    # Use a penalty term that accounts for the cost of evaluation.
    # The base LCB score is: beta * sigma - mu
    # We scale this by a cost-awareness factor that penalizes expensive points
    # but also considers the uncertainty relative to cost.
    
    # Compute the base LCB score
    lcb_score = beta * sigma_safe - mu
    
    # Cost-awareness: Use a softmax-like weighting where the effective score
    # is the LCB score divided by a cost-adjusted penalty.
    # The penalty is cost * (1 + alpha * sigma_safe / (sigma_safe.mean() + 1e-8))
    # This makes expensive points less attractive, but high-uncertainty points
    # get a slight boost to their cost penalty (they're worth more exploration).
    
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean < 1e-8:
        sigma_mean = 1e-8
    
    # Alpha controls how much uncertainty affects the cost penalty
    # Higher alpha means high-uncertainty points are penalized less by cost
    alpha = 0.5
    
    # Cost penalty: higher cost = higher penalty, but high sigma reduces penalty
    cost_penalty = cost * (1.0 + alpha * sigma_safe / (sigma_mean + 1e-8))
    cost_penalty = np.maximum(cost_penalty, 1e-8)
    
    # Final score: LCB score divided by cost penalty
    score = lcb_score / cost_penalty
    
    # Normalize score for stability: subtract mean and scale by std
    score_mean = np.mean(score)
    score_std = np.std(score)
    if score_std < 1e-8:
        score_std = 1e-8
    
    score_normalized = (score - score_mean) / score_std
    
    # Ensure all values are finite
    score_normalized = np.where(np.isfinite(score_normalized), score_normalized, 0.0)
    
    # Check deadline
    if time.time() - t_start > deadline:
        return score_normalized
    
    return score_normalized
