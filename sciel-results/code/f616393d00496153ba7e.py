import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses a hybrid of Expected Improvement (EI) and UCB with adaptive weighting.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Compute UCB score for minimization
    # We want to balance exploitation (low mu) and exploration (high sigma)
    # Score to maximize: kappa * sigma - mu
    
    # Adapt kappa based on f_best progress
    # When f_best is high (early), we want more exploration (higher kappa)
    # When f_best is low (late), we want more exploitation (lower kappa)
    # f_best starts around 25 (5^2) and goes to 0
    f_norm = f_best / (f_best + 5.0)
    kappa = 0.5 + 2.0 * f_norm
    
    ucb = kappa * sigma_safe - mu
    ucb = np.nan_to_num(ucb, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Normalize both scores to [0, 1] range for fair combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    if ucb_max > ucb_min:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Adaptive weight: more EI weight as we converge (f_best decreases)
    # weight_ei goes from 0.3 (early) to 0.9 (late)
    weight_ei = 0.3 + 0.6 * (1.0 - f_norm)
    weight_ucb = 1.0 - weight_ei
    
    # Combine scores
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Final safety check
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    
    return score
