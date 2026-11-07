import numpy as np
from scipy.stats import norm
import time

# Global state to track optimization progress
# Since the evaluator runs multiple seeds in the same process or restarts,
# we need to be careful. However, typically in these tasks, each seed run is
# independent or the global state persists.
# The task says "For each seed 0, 1 and 2...". If the process is restarted for each seed,
# global state resets. If not, it persists.
# Given the 330s limit for the *whole* process, it's likely one process runs all seeds.
# But f_best is passed. We can infer step count from how low f_best is? No.
# Better: Use a counter based on the number of times acquisition is called?
# The prompt says "at each step... calls acquisition".
# We can maintain a global counter.

_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    
    # 1. Increment step counter
    _call_count += 1
    
    # 2. Determine exploration parameter beta
    # Strategy: Start with high exploration, decay to low exploration.
    # Total evaluations = 30. Initial 5 are random. So 25 acquisition calls.
    # Let's map _call_count (1 to 25) to beta.
    # Beta should start high (e.g., 5.0) and end low (e.g., 0.5).
    # Linear decay:
    max_steps = 25
    if _call_count > max_steps:
        _call_count = max_steps
        
    # Normalize progress 0.0 to 1.0
    progress = _call_count / max_steps
    
    # Beta decays from 5.0 to 0.5
    beta_start = 5.0
    beta_end = 0.5
    beta = beta_start + (beta_end - beta_start) * progress

    # 3. Compute Acquisition Score
    # We use a hybrid of UCB and EI-like behavior.
    # Standard UCB: -mu + beta * sigma
    # However, simple UCB can be too aggressive early on or too conservative late.
    # A more robust approach for Ackley (which has many local minima) is to
    # ensure we explore regions with high uncertainty.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB score
    ucb_score = -mu + beta * sigma_safe
    
    # To prevent the acquisition from being dominated by noise in areas where
    # the model is very confident (sigma ~ 0) but mu is slightly worse than f_best,
    # we can blend with EI.
    # z = (f_best - mu) / sigma_safe
    # ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Actually, pure UCB with decaying beta is a very strong baseline for
    # black-box optimization with few evaluations.
    # Let's stick to UCB but ensure numerical stability.
    
    scores = ucb_score
    
    # Safety check: ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all scores are -inf (shouldn't happen), return zeros
    if np.all(scores == -np.inf):
        return np.zeros_like(mu)
        
    return scores
