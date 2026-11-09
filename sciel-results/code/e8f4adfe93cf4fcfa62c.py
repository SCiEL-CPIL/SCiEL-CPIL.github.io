"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit

def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function combining Expected Improvement and Uncertainty.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative and finite, avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute CDF and PDF of standard normal distribution
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    ei_scores = (f_best - mu) * cdf + sigma * pdf
    
    # Normalize EI and Sigma to [0, 1] to balance their magnitudes
    # This allows a simple weighted sum to be effective regardless of the scale of the objective function
    ei_min = np.min(ei_scores)
    ei_max = np.max(ei_scores)
    ei_range = ei_max - ei_min
    if ei_range < 1e-8:
        ei_norm = np.zeros_like(ei_scores)
    else:
        ei_norm = (ei_scores - ei_min) / ei_range
        
    sig_min = np.min(sigma)
    sig_max = np.max(sigma)
    sig_range = sig_max - sig_min
    if sig_range < 1e-8:
        sig_norm = np.zeros_like(sigma)
    else:
        sig_norm = (sigma - sig_min) / sig_range

    # Dynamic weighting:
    # Early in optimization, sigma is generally high. We want to explore.
    # Late in optimization, sigma is lower. We want to exploit.
    # We use the mean sigma as a proxy for "stage of optimization".
    # If mean sigma is high, we increase the weight of the exploration term.
    # We clip the weight to ensure we don't become purely exploratory or purely exploitative.
    
    mean_sigma = np.mean(sigma)
    
    # Heuristic: 
    # If mean_sigma is large (early), weight on exploration (sig_norm) should be higher.
    # If mean_sigma is small (late), weight on exploration should be lower.
    # Let's map mean_sigma to a weight.
    # We assume a typical range for Hartmann 6D standardized GP.
    # A simple linear decay of exploration weight with mean_sigma might not be right if sigma scales with f_best.
    # Let's try a fixed strong exploration bonus first, as suggested by the "inspiration" that worked.
    # The inspiration said: EI + 0.1 * sigma improved score.
    # Let's make it slightly adaptive: 
    # weight = 0.2 if mean_sigma > 1.0 else 0.1
    # Or better, just use the normalized scores and a fixed weight that favors exploration slightly.
    
    # Let's use a weight of 0.3 for exploration (uncertainty) and 0.7 for exploitation (EI).
    # This is a structural change from pure EI (1.0 exploitation).
    
    alpha_exploit = 0.7
    alpha_explore = 0.3
    
    scores = alpha_exploit * ei_norm + alpha_explore * sig_norm
    
    # Clean up non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
