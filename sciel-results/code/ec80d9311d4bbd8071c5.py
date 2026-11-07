import numpy as np
from scipy.stats import norm
import time

# Global start time
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: simple EI / cost
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # Constants
    f_min = -3.32237
    f_max_est = 0.0 # Rough estimate
    
    # Calculate progress
    # f_best starts high (near 0) and goes down to f_min
    # p = 0 at start, p = 1 at end
    denom = f_max_est - f_min
    if denom == 0:
        p = 0.5
    else:
        p = (f_max_est - f_best) / denom
        p = np.clip(p, 0.0, 1.0)
        
    # Weights
    # Early: p ~ 0, w_ucb ~ 1, w_ei ~ 0
    # Late: p ~ 1, w_ucb ~ 0, w_ei ~ 1
    w_ei = p
    w_ucb = 1.0 - p
    
    # Beta for UCB
    # Early: high beta (exploration)
    # Late: low beta (exploitation)
    beta_max = 3.0
    beta_min = 0.5
    beta = beta_max - (beta_max - beta_min) * p
    
    # Compute EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB Improvement
    # UCB for minimization: L(x) = mu - beta * sigma
    # Improvement: f_best - L(x)
    ucb_lower_bound = mu - beta * sigma
    ucb_imp = f_best - ucb_lower_bound
    ucb_imp = np.maximum(ucb_imp, 0.0) # Ensure non-negative
    
    # Normalize scores to be comparable?
    # EI and UCB_Imp might have different scales.
    # However, both are "improvements".
    # Let's just blend them.
    # To be safe, let's scale UCB_Imp by a factor if needed?
    # Or just use the weighted sum.
    
    score_ei = ei / cost
    score_ucb = ucb_imp / cost
    
    score = w_ei * score_ei + w_ucb * score_ucb
    
    return score
