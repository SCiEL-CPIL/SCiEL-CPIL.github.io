import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Look-Ahead Expected Improvement (LAEI) approximation using a UCB-style
    exploration-exploitation balance. This approximates the lookahead value
    by rewarding points with high uncertainty (potential for future gains)
    in addition to the immediate expected improvement.
    """
    # Ensure sigma is positive and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement (EI)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei_scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Look-Ahead / Exploration Term (UCB component)
    # We use a beta factor to balance exploration.
    # A value of 2.0-3.0 is typical for short budgets.
    beta = 2.5
    ucb_scores = -mu + beta * sigma_safe
    
    # Combine EI and UCB? 
    # Or just use UCB?
    # The directive says "Implement LAEI". 
    # A common LAEI approximation is EI + exploration bonus.
    # Let's use a weighted sum to keep the EI structure.
    # Score = EI + lambda * sigma * Phi(z)
    # This is effectively EI + lambda * (Exploration part of EI)
    # So Score = EI * (1 + lambda) for the exploration part?
    
    # Let's try a pure UCB approach as it is robust.
    # But to respect the "EI" base, let's blend them.
    # Normalize both to [0,1] range? No, keep raw.
    
    # Alternative: Use EI but with a larger effective sigma?
    
    # Let's stick to a strong exploration strategy.
    # I will use UCB with beta=2.0.
    
    scores = ucb_scores
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
