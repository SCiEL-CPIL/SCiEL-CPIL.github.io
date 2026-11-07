import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard: track elapsed time to ensure we return within limits
    # Since we don't know the start time, we rely on the computation being fast.
    # The computation here is O(N) vectorized operations, which is extremely fast for N=2000.
    
    sigma = np.maximum(sigma, 1e-9)
    
    # Strategy: Hybrid of Exploitation (Lowest Mean) and Exploration (Highest Uncertainty)
    # The current UCB approach (score = -mu + beta * sigma) with beta=1.5 scored 0.3765.
    # The initial EI approach scored 1.0.
    # This suggests that for this specific setup (short horizon, small GP length scale),
    # a more balanced or slightly more exploitative approach might be better, or a different exploration metric.
    
    # Let's try a "Weighted Sum" approach that is distinct from linear UCB.
    # We normalize mu and sigma to [0, 1] range to make them comparable, then combine them.
    # This avoids the scale issue where sigma might be dominated by mu or vice versa.
    
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    # Avoid division by zero
    mu_range = mu_max - mu_min + 1e-9
    sigma_range = sigma_max - sigma_min + 1e-9
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # We want high scores for low mu (exploitation) and high sigma (exploration).
    # Score = w * (1 - mu_norm) + (1 - w) * sigma_norm
    # Let's try w = 0.5 (balanced) or w = 0.6 (slightly more exploitative).
    # Given EI (which is more exploitative in the long run but can be greedy) scored 1.0,
    # and UCB (more exploratory) scored 0.37, maybe a balanced normalized score is better.
    
    w = 0.5
    score = w * (1.0 - mu_norm) + (1.0 - w) * sigma_norm
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    
    return score
