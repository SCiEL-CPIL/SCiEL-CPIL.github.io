import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Hybrid EI/UCB acquisition for minimizing Michalewicz 2D.
    
    Uses Expected Improvement (EI) to target regions likely to improve
    the best observed value, combined with a UCB-style exploration bonus
    to maintain exploration in the multimodal landscape.
    
    score = EI + alpha * sigma
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Sanitize inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to prevent numerical issues in exp/cdf
    z = np.clip(z, -10.0, 10.0)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Exploration bonus
    # A moderate alpha helps escape local minima in multimodal functions
    alpha = 0.5
    score = EI + alpha * sigma
    
    # Guarantee finiteness
    if not np.all(np.isfinite(score)):
        score = np.where(np.isfinite(score), score, -mu)
        if not np.all(np.isfinite(score)):
            score = -np.where(np.isfinite(mu), mu, f_best)

    # Safety: if this ever becomes slow, fall back to fast exploitation
    if time.time() - t0 > 30.0:
        return -np.where(np.isfinite(mu), mu, f_best)

    return score
