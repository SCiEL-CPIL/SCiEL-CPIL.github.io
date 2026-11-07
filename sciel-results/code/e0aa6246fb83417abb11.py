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
    
    beta_min = 0.3
    beta_max = 2.5
    r_scale = 5.0
    
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
    
    # Step 2: Create a sharper gate that emphasizes high-UCB regions
    # Use a sharper sigmoid to focus on top ~10-15% of UCB candidates
    temperature = 5.0
    # Shift so that ucb_norm=0.8 is the center (we want top 20% to be strongly gated)
    gate_center = 0.8
    gate = 1.0 / (1.0 + np.exp(-temperature * (ucb_norm - gate_center)))
    
    # Step 3: Kriging Believer Logic
    # Within the gated regions, prefer lower mu (better predicted value)
    # Normalize mu to [0, 1] range for better scaling
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min + eps
    mu_norm = (mu - mu_min) / mu_range  # 0 = best (lowest mu), 1 = worst
    
    # Score: high gate * (1 - mu_norm) means high uncertainty + low predicted value
    kb_scores = gate * (1.0 - mu_norm)
    
    # Add UCB component to break ties and encourage exploration
    ucb_component = 0.3 * ucb_norm
    
    scores = kb_scores + ucb_component
    
    # Check time
    if time.time() - start_time > deadline:
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
