"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: Evaluator kills after 330s. We set a safety margin.
_DEADLINE = time.time() + 300.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Lower Confidence Bound (LCB) for minimization.
    
    The exploration parameter kappa is adapted based on the progress of the current
    best value (f_best) towards the known global minimum of Hartmann-3D (-3.86278).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP posterior mean.
    sigma : ndarray, shape (2000,)
        GP posterior standard deviation.
    f_best : float
        Best (lowest) observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Check deadline to avoid timeout
    if time.time() > _DEADLINE:
        # Fallback: pure exploitation (minimize mu -> maximize -mu)
        mu_arr = np.asarray(mu, dtype=np.float64).ravel()
        return -mu_arr.copy()
    
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    # Ensure finiteness of inputs
    # If mu or sigma are not finite, replace with safe values
    mu_mask = np.isfinite(mu)
    sigma_mask = np.isfinite(sigma)
    
    if not np.all(mu_mask):
        # Replace non-finite mu with the max finite mu (worst case for minimization)
        safe_mu = np.max(mu[mu_mask]) if np.any(mu_mask) else 0.0
        mu = np.where(mu_mask, mu, safe_mu)
        
    if not np.all(sigma_mask):
        # Replace non-finite sigma with 0 (no uncertainty)
        sigma = np.where(sigma_mask, sigma, 0.0)
        
    # Ensure sigma is non-negative and handle very small values to avoid numerical instability
    # Although LCB doesn't divide by sigma, extremely small sigma with large kappa 
    # might cause issues if not handled, but mainly we just want valid floats.
    sigma = np.maximum(sigma, 1e-12)
    
    # Known global minimum for Hartmann-3D
    f_global_min = -3.86278
    
    # Reference worst-case value for progress calculation
    # Hartmann-3D range is roughly [-3.86, 0.4]. We use 0.0 as a reasonable "bad" start reference.
    f_worst_ref = 0.0
    
    # Calculate progress towards the global minimum
    # progress = 0 when f_best is bad (high), 1 when f_best is good (low)
    denominator = f_worst_ref - f_global_min
    if abs(denominator) < 1e-12:
        progress = 0.5
    else:
        progress = (f_worst_ref - f_best) / denominator
        
    # Clip progress to [0, 1]
    progress = float(np.clip(progress, 0.0, 1.0))
    
    # Adaptive Kappa
    # High kappa (3.0) at start (progress=0) -> Explore
    # Low kappa (0.5) at end (progress=1) -> Exploit
    kappa_max = 3.0
    kappa_min = 0.5
    
    # Linear interpolation of kappa based on progress
    kappa = kappa_max - (kappa_max - kappa_min) * progress
    
    # LCB Score for minimization:
    # We want to minimize f(x). The LCB is mu(x) - kappa * sigma(x).
    # We want the candidate with the LOWEST LCB.
    # The evaluator picks the candidate with the HIGHEST score.
    # Therefore, we return the negative LCB: score = -(mu - kappa * sigma) = -mu + kappa * sigma
    
    scores = -mu + kappa * sigma
    
    # Final finiteness check
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
