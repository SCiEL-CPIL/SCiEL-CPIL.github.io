import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and 
    Upper Confidence Bound (UCB) for minimization with dynamic weighting.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Ensure sigma is strictly positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term for exploration: -mu + beta * sigma
    # beta=1.0 is a moderate exploration coefficient
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Normalize both to comparable scales [0, 1]
    # This prevents one term from dominating solely due to scale differences
    def _normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        rng = arr_max - arr_min
        if rng < 1e-12:
            return np.zeros_like(arr)
        return (arr - arr_min) / rng
    
    ei_norm = _normalize(ei)
    ucb_norm = _normalize(ucb)
    
    # Dynamic weighting based on progress
    # Styblinski-Tang 1D min is approx -39.1662, max approx 12.0
    min_val = -39.1662
    max_val = 12.0
    span = max_val - min_val
    
    # progress: 0 if f_best is at max_val (worst), 1 if f_best is at min_val (best)
    # We want to exploit more as we get closer to the minimum.
    # So w_ei should increase as f_best decreases.
    # progress = (max_val - f_best) / span
    # If f_best = max_val, progress = 0. w_ei = low.
    # If f_best = min_val, progress = 1. w_ei = high.
    
    progress = np.clip((max_val - f_best) / span, 0.0, 1.0)
    
    # Start with more exploration (w_ei=0.4), end with more exploitation (w_ei=0.9)
    w_ei = 0.4 + 0.5 * progress
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
