import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time
_start_time = time.time()
_DEADLINE = 325.0  # 5 second safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Time guard
    current_time = time.time()
    if current_time - _start_time > _DEADLINE:
        # Return a safe, valid result (e.g., uniform random or zeros)
        # Must return 2000 finite scores.
        # Using a fixed small positive number or random.
        # Random might be better to break symmetry if stuck.
        return np.random.rand(len(mu)) * 1e-6 + 1e-9

    # Ensure arrays are 1D and float
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Avoid division by zero in cost, though cost >= 1
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + 1e-9)
    # EI formula
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    score_ei = ei / cost_safe
    
    # 2. Compute Cost-Aware UCB (Upper Confidence Bound for minimization)
    # We want to minimize f. 
    # LCB = mu - kappa * sigma
    # We want to maximize -LCB = -mu + kappa * sigma
    kappa = 1.5  # Tuned for exploration in multimodal functions
    score_ucb = (-mu + kappa * sigma) / cost_safe
    
    # 3. Hybrid: Use log-sum-exp for a smooth max of the two scores
    # This avoids the kink in np.maximum and may lead to smoother optimization behavior
    # log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    a = score_ei
    b = score_ucb
    m = np.maximum(a, b)
    # Handle case where both are very negative (underflow in exp)
    # If m is very negative, the sum is essentially 0, so log-sum-exp is -inf
    # But we want a finite score.
    # Let's use a stable log-sum-exp formula.
    # log(exp(a) + exp(b)) = m + log(1 + exp(a-m) + exp(b-m))
    # If m is -inf, this is problematic.
    # Let's just use np.logaddexp(a, b) which is stable.
    scores = np.logaddexp(a, b)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
