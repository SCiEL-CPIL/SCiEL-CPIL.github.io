"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit
_eval_count = 0


def acquisition(mu, sigma, f_best):
    """
    Multi-objective acquisition function combining Expected Improvement (EI)
    and predictive uncertainty with dynamic weighting.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    global _eval_count
    
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Increment evaluation counter
    _eval_count += 1
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    from scipy.stats import norm
    
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Normalize EI to [0, 1] range
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max - ei_min > 1e-10:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    # Normalize sigma to [0, 1] range
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    if sigma_max - sigma_min > 1e-10:
        sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Dynamic weight: increase exploration weight as we have more evaluations
    # Start with high exploration (weight on sigma) and shift toward exploitation (weight on EI)
    # Use a smooth transition: weight_ei starts low and increases with evaluations
    # After 5 initial points, we have _eval_count - 5 additional evaluations
    n_additional = max(_eval_count - 5, 0)
    
    # Weight for EI increases from ~0.3 to ~0.7 as evaluations progress
    # Use a sigmoid-like function centered around 15 evaluations
    weight_ei = 0.3 + 0.4 * (1.0 / (1.0 + np.exp(-(n_additional - 15) / 5.0)))
    weight_sigma = 1.0 - weight_ei
    
    # Combine the two objectives
    scores = weight_ei * ei_norm + weight_sigma * sigma_norm
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
