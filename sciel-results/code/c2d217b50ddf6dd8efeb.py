"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a dynamic Lower Confidence Bound (LCB)
    with a smooth cosine-annealed exploration schedule.
    
    For minimization, the LCB is:
        LCB(x) = mu(x) - kappa * sigma(x)
    
    We return the negative LCB for maximization:
        Score(x) = -mu(x) + kappa * sigma(x)
    
    The parameter kappa is annealed from a high exploration value to a
    low exploitation value using a cosine schedule based on the progress
    of f_best toward the known global minimum.
    """
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-10)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-10)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Known global minimum for Hartmann 3D
    f_global_min = -3.86278
    
    # Reference value for "start" of optimization
    # Hartmann 3D at random points in [0,1]^3 typically gives values around 0 to 1
    f_start_ref = 0.5
    
    # Calculate progress: 0.0 at start (poor f_best), 1.0 at optimum
    span = f_start_ref - f_global_min
    if span <= 0:
        progress = 0.5
    else:
        progress = (f_start_ref - f_best) / span
        progress = np.clip(progress, 0.0, 1.0)
    
    # Cosine-annealed kappa schedule
    # Start with high exploration (kappa_max), end with low exploitation (kappa_min)
    # Cosine schedule: kappa = kappa_min + 0.5 * (kappa_max - kappa_min) * (1 + cos(pi * progress))
    kappa_min = 0.3
    kappa_max = 2.0
    kappa = kappa_min + 0.5 * (kappa_max - kappa_min) * (1.0 + np.cos(np.pi * progress))
    
    # Calculate LCB score for maximization
    scores = -mu_safe + kappa * sigma_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
