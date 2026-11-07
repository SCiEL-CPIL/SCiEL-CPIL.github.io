"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    
    Uses a hybrid of Expected Improvement (EI) and Probability of Improvement (PI)
    with a small exploration bonus. The blend balances exploitation (PI) with
    improvement magnitude (EI) to avoid heavy-tailed exploration in high dimensions.
    """
    # Track time to ensure we finish well within the 300s limit
    t0 = time.time()
    
    # Ensure inputs are proper numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite inputs gracefully
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        return np.zeros(len(mu), dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Compute Phi(z) and phi(z) using scipy.stats for numerical stability
    phi_z = norm.cdf(z)
    phi_pdf = norm.pdf(z)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * phi_z + sigma * phi_pdf
    
    # Probability of Improvement for minimization:
    # PI = Phi(z) where z = (f_best - mu) / sigma
    pi = phi_z
    
    # Normalize EI to [0, 1] range for blending with PI
    # Use a soft normalization based on the max EI
    max_ei = np.max(ei)
    if max_ei > 1e-12:
        ei_norm = ei / max_ei
    else:
        ei_norm = np.zeros_like(ei)
    
    # Blend: 0.4 * PI + 0.6 * EI_norm
    # This gives more weight to probability of improvement while still
    # considering the magnitude of expected improvement
    blended = 0.4 * pi + 0.6 * ei_norm
    
    # Add a small exploration term to prevent pure exploitation
    exploration_term = 0.05 * sigma / (sigma + 0.1)
    
    scores = blended + exploration_term
    
    # Ensure all returned scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline (should be very fast, O(N) computation)
    elapsed = time.time() - t0
    if elapsed > 280.0:
        # Return current scores as fallback
        pass
    
    return scores
# EVOLVE-BLOCK-END
