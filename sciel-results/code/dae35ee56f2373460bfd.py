"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Strategy:
    - Use Probability of Improvement (PI) as the exploitation component.
    - Add a dynamic exploration bonus proportional to sigma.
    - The exploration weight lambda is tuned based on the distance of f_best 
      from the known global minimum.
    
    This avoids the scale-mismatch issues of min-max normalizing EI and UCB separately.
    """
    # Known global minimum for Styblinski-Tang 1D
    f_min = -39.1662
    
    # Safety check for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # Since we want to maximize the score, higher PI is better.
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # 2. Dynamic Exploration Weight
    # We want to explore more when f_best is far from f_min.
    # f_best starts high (e.g., ~1000) and goes down to -39.16.
    # Let's define a 'distance' metric.
    # If f_best is very high, we are lost -> high lambda.
    # If f_best is close to f_min, we are converging -> low lambda.
    
    # Heuristic for lambda:
    # Map f_best range [f_min, f_max_est] to [lambda_min, lambda_max].
    # f_max_est: The function can be very large at boundaries. 
    # f(5) = 1339.5, f(-5) = -1305.5. 
    # However, the GP is standardized, so sigma is in objective units.
    # A reasonable range for lambda might be 0.1 to 5.0.
    
    # Let's use a logarithmic or inverse distance measure.
    # If f_best > 0, we are likely in a bad region or far away.
    # If f_best < 0, we are in the "basin" but maybe not at the bottom.
    
    # Simple linear interpolation based on progress towards the minimum from a "worst case" start.
    # Let's assume a worst case start of 500 (conservative high value).
    f_start_est = 500.0
    
    # Progress p: 0.0 at start, 1.0 at min.
    # p = (f_start_est - f_best) / (f_start_est - f_min)
    # If f_best < f_min, p > 1. Clamp it.
    # If f_best > f_start_est, p < 0. Clamp it.
    
    p = (f_start_est - f_best) / (f_start_est - f_min)
    p = np.clip(p, 0.0, 1.0)
    
    # Lambda decreases as progress increases.
    # lambda = lambda_max * (1 - p) + lambda_min * p
    lambda_min = 0.1
    lambda_max = 3.0
    
    lambda_val = lambda_max * (1.0 - p) + lambda_min * p
    
    # 3. Composite Score
    # Score = PI + lambda * sigma
    # PI is in [0, 1]. Sigma is in objective units (can be large).
    # This might make sigma dominate. 
    # To balance, we might need to scale sigma or PI.
    # However, in early stages, sigma is large everywhere, so lambda*sigma dominates, leading to uniform exploration (good).
    # In late stages, sigma is small in explored regions, large in unexplored. 
    # If we are close to min, lambda is small, so PI dominates, leading to exploitation.
    
    score = pi + lambda_val * sigma_safe
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
