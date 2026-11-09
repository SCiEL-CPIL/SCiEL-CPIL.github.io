import numpy as np
from scipy.stats import norm

def _expected_improvement(mu, sigma, f_best):
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _ucb_score(mu, sigma, f_best, beta):
    sigma_safe = np.maximum(sigma, 1e-8)
    return -mu + beta * sigma_safe

def acquisition(mu, sigma, f_best):
    # Estimate progress: 
    # Assume initial sigma is ~1.0 (standardized targets)
    # As GP learns, sigma decreases.
    mean_sigma = np.mean(sigma)
    # Clip progress to [0, 1]
    # If mean_sigma is 1.0, progress = 0. If mean_sigma is 0.0, progress = 1.
    progress = 1.0 - mean_sigma
    progress = np.clip(progress, 0.0, 1.0)
    
    # Weight for EI: increases with progress
    # Early: 0.5 (balanced)
    # Late: 1.0 (exploit)
    w_ei = 0.5 + 0.5 * progress
    
    # Beta for UCB: decreases with progress?
    # Or keep it constant?
    beta = 0.5
    
    ei = _expected_improvement(mu, sigma, f_best)
    ucb = _ucb_score(mu, sigma, f_best, beta)
    
    # Normalize EI and UCB to [0, 1] to combine them fairly
    # EI is always >= 0.
    ei_min, ei_max = np.min(ei), np.max(ei)
    if ei_max - ei_min > 1e-8:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
        
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    if ucb_max - ucb_min > 1e-8:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
        
    score = w_ei * ei_norm + (1 - w_ei) * ucb_norm
    
    return score
