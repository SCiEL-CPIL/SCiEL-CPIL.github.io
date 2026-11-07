"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Handle edge cases
    if len(mu) == 0:
        return np.array([])
    
    # Ensure finite values and proper types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    n = len(mu)
    
    # Knowledge Gradient (KG) Acquisition
    # KG(x) = f_best - E[min(f_best, f(x))]
    #      = f_best - (f_best * P(f(x) > f_best) + E[f(x) | f(x) < f_best] * P(f(x) < f_best))
    #      = f_best - (f_best * (1 - Phi(z)) + (mu - sigma * phi(z)/Phi(z)) * Phi(z))
    # where z = (f_best - mu) / sigma
    
    # Calculate z-scores
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    # norm.cdf handles large values well, but phi(z)/Phi(z) can be unstable if Phi(z) is tiny
    # We clip z to [-50, 50] for safety, though in practice values are usually within [-10, 10]
    z_clipped = np.clip(z, -50, 50)
    
    # Compute Phi(z) and phi(z)
    Phi_z = norm.cdf(z_clipped)
    phi_z = norm.pdf(z_clipped)
    
    # To avoid division by zero in phi(z)/Phi(z), add a small epsilon to Phi_z
    # If Phi_z is very small, the term (mu - sigma * phi/Phi) * Phi is effectively 0 anyway
    # because Phi is tiny. But to be safe:
    Phi_z_safe = np.maximum(Phi_z, 1e-300)
    
    # Calculate the expected minimum
    # E[min(f_best, f(x))] = f_best * (1 - Phi_z) + (mu - sigma * phi_z / Phi_z) * Phi_z
    # Note: E[f(x) | f(x) < f_best] = mu - sigma * phi(z) / Phi(z)
    
    # Optimization:
    # KG = f_best - [f_best * (1 - Phi_z) + (mu - sigma * phi_z / Phi_z) * Phi_z]
    # KG = f_best - [f_best - f_best * Phi_z + mu * Phi_z - sigma * phi_z]
    # KG = f_best * Phi_z - mu * Phi_z + sigma * phi_z
    # KG = Phi_z * (f_best - mu) + sigma * phi_z
    
    # This simplified form is numerically more stable and faster.
    # Let's verify:
    # If f_best >> mu (z is large positive), Phi_z ~ 1, phi_z ~ 0.
    # KG ~ 1 * (f_best - mu) + 0 = f_best - mu. Correct (high improvement).
    # If f_best << mu (z is large negative), Phi_z ~ 0, phi_z ~ 0.
    # KG ~ 0. Correct (low improvement).
    # If f_best == mu (z = 0), Phi_z = 0.5, phi_z = 0.3989.
    # KG = 0.5 * 0 + sigma * 0.3989 = 0.3989 * sigma. Correct.
    
    kg_scores = Phi_z * (f_best - mu) + sigma * phi_z
    
    # Ensure all scores are finite
    kg_scores = np.where(np.isfinite(kg_scores), kg_scores, 0.0)
    
    # KG is always non-negative. 
    # We return higher scores for better candidates.
    return kg_scores
# EVOLVE-BLOCK-END
