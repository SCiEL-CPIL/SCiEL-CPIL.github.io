import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Hybrid EI/UCB acquisition for minimizing Michalewicz 2D.
    
    Uses Expected Improvement (EI) as the primary driver to target 
    regions likely to improve the best observed value, combined with 
    a small uncertainty bonus (sigma) to maintain exploration in 
    multimodal landscapes.
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Sanitize inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values in CDF/PDF calculations
    # norm.cdf and norm.pdf handle large values well, but clipping prevents overflow in intermediate steps if any
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Add a small exploration bonus to prevent premature convergence
    # in the multimodal Michalewicz landscape.
    # The coefficient 0.2 is a heuristic balance.
    exploration_bonus = 0.2 * sigma
    
    score = ei + exploration_bonus
    
    # Ensure finiteness
    if not np.all(np.isfinite(score)):
        score = np.where(np.isfinite(score), score, 0.0)
        # If still not finite, fall back to exploitation (negative mu)
        if not np.all(np.isfinite(score)):
            score = -np.where(np.isfinite(mu), mu, f_best)
    
    # Safety deadline guard
    if time.time() - t0 > 30.0:
        return -np.where(np.isfinite(mu), mu, f_best)
        
    return score
