"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a smooth soft-max of EI and UCB per unit cost.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Check deadline
    if time.time() - _START_TIME > _TIMEOUT:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # 1. Expected Improvement (EI) per unit cost
    # Standard EI for minimization: (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in exp/cdf
    z_clipped = np.clip(z, -10.0, 10.0)
    
    ei_term = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei_term = np.maximum(ei_term, 0.0)
    
    score_ei = ei_term / cost_safe
    
    # 2. Upper Confidence Bound (UCB) per unit cost
    # We want to explore high uncertainty. 
    # Score should be high when mu is low (good) or sigma is high (uncertain).
    # UCB value to minimize: mu - beta * sigma
    # Acquisition score (maximize): -(mu - beta * sigma) = -mu + beta * sigma
    # Normalize relative to f_best to make scale comparable?
    # The prompt says "higher = evaluate next".
    # Let's use (f_best - mu + beta * sigma) / cost
    # This rewards points that are below f_best (exploit) or have high uncertainty (explore).
    beta = 2.0 # Exploration parameter
    score_ucb = (f_best - mu + beta * sigma_safe) / cost_safe
    
    # 3. Smooth Aggregation (Soft-Max)
    # Instead of hard max, use log-sum-exp to create a smooth surrogate for max(EI, UCB).
    # This allows the gradient of the acquisition function to be non-zero even where one term dominates,
    # potentially allowing the search to "slide" along ridges or escape local maxima of the hard-max landscape.
    
    # Temperature parameter for soft-max. Lower = closer to hard max. Higher = smoother (more average-like).
    # A value around 0.1 to 0.5 is typical. Let's try 0.2.
    temp = 0.2
    
    # Log-sum-exp trick for stability
    # max_score = max(score_ei, score_ucb)
    # scores = max_score + temp * log(exp((score_ei - max_score)/temp) + exp((score_ucb - max_score)/temp))
    
    max_score = np.maximum(score_ei, score_ucb)
    
    # Handle cases where scores are -inf or 0 to avoid log(0) or overflow
    # If both are 0, result is 0.
    # We compute: temp * log( exp((ei-max)/temp) + exp((ucb-max)/temp) ) + max
    # Let a = (ei - max)/temp, b = (ucb - max)/temp. Both <= 0.
    # log(exp(a) + exp(b)) = log(exp(a) * (1 + exp(b-a))) = a + log(1 + exp(b-a))
    # Since max is one of them, one of a,b is 0. The other is <= 0.
    
    a = (score_ei - max_score) / temp
    b = (score_ucb - max_score) / temp
    
    # Clip a and b to avoid underflow in exp (though exp of negative large is 0, which is fine)
    # But we need to handle the case where max_score is -inf or 0.
    # If max_score is 0, a and b are 0 (if ei=ucb=0) or -inf.
    
    # Safe calculation
    # If max_score <= 0, scores are likely 0 or negative. 
    # EI is >= 0. UCB can be negative if mu > f_best + beta*sigma.
    # We want scores to be 0 if no improvement is expected.
    
    # Let's use a simpler stable soft-max implementation
    # S = temp * log( sum( exp(score_i / temp) ) )
    # To prevent overflow, subtract max:
    # S = max_score + temp * log( sum( exp((score_i - max_score) / temp) ) )
    
    # If max_score is -inf, return 0.
    # If max_score is finite:
    exp_a = np.exp(a)
    exp_b = np.exp(b)
    
    log_sum = np.log(exp_a + exp_b)
    
    smooth_scores = max_score + temp * log_sum
    
    # Ensure non-negative (acquisition should be 0 if no improvement)
    # Actually, UCB can be negative. EI is non-negative.
    # If both are negative, smooth score is negative. We should probably floor at 0?
    # Standard EI is floored at 0. UCB is not.
    # If we floor at 0, we lose the "penalty" for bad points in UCB? No, UCB is a score to maximize.
    # If UCB is negative, it's worse than 0.
    # Let's keep the raw smooth score but ensure it's finite.
    
    # However, if max_score is very negative, exp_a and exp_b will be 0, log(0) is -inf.
    # This happens if scores are -inf.
    # We initialized sigma_safe and cost_safe, so scores shouldn't be -inf unless input is.
    
    # Replace non-finite with 0
    smooth_scores = np.where(np.isfinite(smooth_scores), smooth_scores, 0.0)
    
    # Floor at 0? 
    # If EI=0 and UCB=-1, softmax is close to -1. 
    # If we floor at 0, we treat it as "neutral". 
    # Usually, we want to prioritize points with positive expected gain.
    # Let's floor at 0 to be safe, similar to EI.
    smooth_scores = np.maximum(smooth_scores, 0.0)
    
    return smooth_scores
# EVOLVE-BLOCK-END
