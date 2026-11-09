import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    
    Uses an adaptive strategy that blends Expected Improvement (EI) and
    Posterior Mean Minimization (PMM) based on the current state of the search.
    Early on (high f_best), EI provides a good balance of exploration and exploitation.
    Late on (low f_best), PMM aggressively exploits the GP's predictive mean to
    fine-tune the location of the minimum.
    
    A smooth blending factor based on f_best transitions between the two.
    """
    start_time = time.time()
    deadline = start_time + 5.0  # Safety margin

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Z score for EI
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (f_best - mu) / sigma
    
    # Compute EI
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Clip Z to avoid numerical issues
    Z_clipped = np.clip(Z, -10, 10)
    
    phi_Z = norm.pdf(Z_clipped)
    Phi_Z = norm.cdf(Z_clipped)
    
    EI = (f_best - mu) * Phi_Z + sigma * phi_Z
    EI = np.nan_to_num(EI, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Check deadline
    if time.time() > deadline:
        return EI

    # PMM score: -mu (higher is better, since we want to minimize mu)
    PMM = -mu
    PMM = np.nan_to_num(PMM, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Blending factor: 
    # When f_best is large (early), we want to use EI more (weight_ei close to 1)
    # When f_best is small (late), we want to use PMM more (weight_ei close to 0)
    # Use a smooth transition: weight_ei = 1 / (1 + f_best)
    # When f_best = 10, weight_ei ~ 0.09 (mostly PMM? No, that's wrong)
    # Let's invert: weight_ei = f_best / (1 + f_best)
    # When f_best = 10, weight_ei ~ 0.91 (mostly EI)
    # When f_best = 1, weight_ei ~ 0.5 (50/50)
    # When f_best = 0.1, weight_ei ~ 0.09 (mostly PMM)
    # When f_best = 0.01, weight_ei ~ 0.01 (almost all PMM)
    
    weight_ei = f_best / (1.0 + f_best)
    weight_pmm = 1.0 - weight_ei
    
    # Normalize EI and PMM to comparable scales before blending
    # EI is always >= 0, PMM can be negative
    # Let's shift PMM so its minimum is 0
    PMM_min = np.min(PMM)
    PMM_shifted = PMM - PMM_min
    PMM_shifted = np.maximum(PMM_shifted, 0.0)
    
    # Scale both to [0, 1] range
    EI_max = np.max(EI)
    PMM_max = np.max(PMM_shifted)
    
    if EI_max > 0:
        EI_norm = EI / EI_max
    else:
        EI_norm = EI
    
    if PMM_max > 0:
        PMM_norm = PMM_shifted / PMM_max
    else:
        PMM_norm = PMM_shifted
    
    # Blend
    scores = weight_ei * EI_norm + weight_pmm * PMM_norm
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
