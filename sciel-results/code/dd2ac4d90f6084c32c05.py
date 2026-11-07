"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Start with a deadline guard: 330s wall-clock limit, use 300s for safety
    t_start = time.time()
    deadline = 300.0  # seconds, leaving 30s margin

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        # Return uniform scores as fallback
        return np.ones_like(mu)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) acquisition function
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite output
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Check time
    elapsed = time.time() - t_start
    if elapsed > deadline:
        return ei
    
    # The EI is a good baseline. Let's consider if we can improve it.
    # The task says "higher meaning evaluate next". EI is standard for this.
    # One improvement: use a small exploration term or adjust the exploitation/exploration trade-off.
    # However, the seed program already uses EI and scores 1.0.
    
    # Let's try a variant: Lower Bound (LCB) or a combination.
    # But the directive says "KEEP this design and investigate improvements within its family."
    # The family is EI-based acquisition.
    
    # Potential improvement: Handle the case where sigma is very small more carefully.
    # Or, use a slightly different formulation that might be more numerically stable.
    
    # Another idea: The task mentions "standardized targets" in the GP. 
    # mu and sigma are in objective units. EI should work directly.
    
    # Let's check if there's a better constant or a slight modification.
    # Sometimes, adding a small epsilon to f_best or adjusting the z-score helps.
    
    # For now, the standard EI is the most robust choice.
    # We'll return the EI scores.
    
    return ei
# EVOLVE-BLOCK-END
