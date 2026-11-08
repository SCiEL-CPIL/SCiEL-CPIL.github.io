import numpy as np
from scipy.stats import norm
import time

# Module-level cache for timing guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize timing if not set
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 330s wall-clock limit, leave 5s margin
    if time.time() - _start_time > 325:
        # Return simple scores if we're about to timeout
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        cost = np.asarray(cost)
        # Simple UCB-based fallback
        scores = sigma / (cost + 1e-8) - mu
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Safe sigma for z-score calculation
    eps = 1e-10
    safe_sigma = np.maximum(sigma, eps)
    
    # --- Strategy 1: Cost-Adjusted Expected Improvement (EI) ---
    z = (f_best - mu) / safe_sigma
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    c_ei = ei / cost
    
    # --- Strategy 2: Cost-Adjusted UCB (Upper Confidence Bound) ---
    # For minimization, we want low mu and high uncertainty
    # UCB = mu - beta * sigma (we minimize this, so score = -UCB = beta*sigma - mu)
    # But we also want to balance exploration vs exploitation
    # Use a moderate beta
    beta = 1.5
    ucb_score = beta * sigma - mu
    c_ucb = ucb_score / cost
    
    # --- Strategy 3: Pure Exploitation with cost adjustment ---
    # Score based on how much better mu is than f_best, divided by cost
    # This rewards points that are likely to improve upon f_best
    improvement = f_best - mu
    # Only positive improvements matter
    improvement = np.maximum(improvement, 0)
    c_exploit = improvement / cost
    
    # --- Strategy 4: Probability of Improvement (PI) cost-adjusted ---
    # P(mu < f_best) = Phi((f_best - mu) / sigma)
    pi_prob = norm.cdf(z_clipped)
    c_pi = pi_prob / cost
    
    # Combine strategies with weighted average
    # Weight EI highest, then UCB, then PI, then exploitation
    w_ei = 0.4
    w_ucb = 0.3
    w_pi = 0.2
    w_exploit = 0.1
    
    scores = w_ei * c_ei + w_ucb * c_ucb + w_pi * c_pi + w_exploit * c_exploit
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
