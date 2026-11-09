import numpy as np
import time
from scipy.special import erf

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    
    # Compute normal PDF and CDF
    phi_z = np.exp(-0.5 * z_clipped ** 2) / np.sqrt(2 * np.pi)
    cdf_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PoI)
    poi = cdf_z
    
    # Blend EI with PoI-based confidence factor
    # (1 + PoI) amplifies scores for points likely to improve
    blended_ei = ei * (1.0 + poi)
    
    # Cost-aware normalization
    # Use log cost to soften the penalty for higher costs
    # This is a structural change from power-law to log-based cost penalty
    cost_factor = np.log(1.0 + cost_safe)
    
    # Cost-adjusted scores
    blended_ei_score = blended_ei / cost_factor
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(blended_ei_score), blended_ei_score, 0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return scores
    
    return scores
