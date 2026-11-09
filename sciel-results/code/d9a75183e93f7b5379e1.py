import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) for minimization.
    
    The weight shifts from exploration (UCB) to exploitation (EI)
    as f_best approaches the minimum (0).
    """
    eps = 1e-9
    sigma_safe = sigma + eps
    
    # 1. Expected Improvement (Exploitation)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Upper Confidence Bound (Exploration)
    # For minimization, we maximize -mu + beta * sigma
    beta = 2.0
    ucb = -mu + beta * sigma
    
    # 3. Adaptive Weighting
    # Determine progress: if f_best is high, we need to explore.
    # If f_best is low, we need to exploit.
    # Scale: 5.0 corresponds to a "medium" regret.
    # f_best in [0, 25].
    # If f_best >= 5, t=1 (Pure UCB).
    # If f_best <= 0, t=0 (Pure EI).
    t = np.clip(f_best / 5.0, 0.0, 1.0)
    
    w_explore = t
    w_exploit = 1.0 - t
    
    score = w_exploit * ei + w_explore * ucb
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
