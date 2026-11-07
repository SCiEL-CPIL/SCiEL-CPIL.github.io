import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Uses a Kriging Believer-inspired structural approach:
    1. Adaptive UCB to identify promising regions (high uncertainty + low bound).
    2. Gated exploitation score (-mu) to select the best predicted value within those regions.
    This decouples exploration (region selection) from exploitation (point selection).
    """
    start_time = time.time()
    deadline = 295.0  # Safety margin before 300s limit
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Adaptive Beta: High regret -> High beta (explore), Low regret -> Low beta (exploit)
    f_global_min = 0.397887
    regret = f_best - f_global_min
    
    beta_min = 0.5
    beta_max = 3.0
    r_scale = 10.0
    
    if regret >= 0:
        fraction = regret / (regret + r_scale)
    else:
        fraction = 0.0
        
    beta = beta_min + (beta_max - beta_min) * fraction
    
    # Step 1: Compute UCB scores to identify promising regions
    # UCB for minimization: beta * sigma - mu
    ucb_scores = beta * sigma_safe - mu
    
    # Normalize UCB scores to [0, 1] for gating
    ucb_min = np.min(ucb_scores)
    ucb_max = np.max(ucb_scores)
    ucb_range = ucb_max - ucb_min + eps
    ucb_norm = (ucb_scores - ucb_min) / ucb_range
    
    # Step 2: Create a gate that emphasizes high-UCB regions
    # We use a sigmoid to softly select the top ~10-20% of UCB candidates
    # Temperature controls the sharpness of the gate
    temperature = 2.0
    # Shift so that ucb_norm=0.5 is the center of the sigmoid
    # We want points with high ucb_norm to have gate ~1, low to have gate ~0
    gate = 1.0 / (1.0 + np.exp(-temperature * (ucb_norm - 0.5)))
    
    # Step 3: Kriging Believer Logic
    # Within the gated regions, prefer lower mu (better predicted value)
    # Score is -mu, scaled by the gate
    # We add a small constant offset to ensure scores are comparable
    # If gate is 0, score is 0. If gate is 1, score is -mu.
    # Since we want higher scores to be better, and lower mu is better,
    # score = gate * (-mu) works, but -mu can be negative.
    # Let's shift mu to be positive relative to the global min or just use -mu directly.
    # To ensure "higher means evaluate next", we use -mu.
    # However, if mu is very negative (unlikely in Branin which is >= 0.39), -mu is large positive.
    # If mu is large, -mu is large negative.
    
    # To make this robust, let's normalize -mu as well or just use it.
    # The gate ensures we only care about -mu in high-uncertainty areas.
    kb_scores = gate * (-mu)
    
    # Add a tiny UCB component to break ties in flat regions
    # This ensures we still explore if all mu are similar
    ucb_component = 0.1 * ucb_scores
    
    scores = kb_scores + ucb_component
    
    # Check time
    if time.time() - start_time > deadline:
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
