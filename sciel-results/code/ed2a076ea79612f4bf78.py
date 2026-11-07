"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 315  # 315s limit, 15s margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid approach:
    - Exploitation: expected improvement (EI) for points close to current best
    - Exploration: UCB term for uncertain regions
    - Cost normalization: divide by cost to prefer cheap evaluations
    
    The key structural change from pure UCB is using a proper EI-based
    exploitation term that is more targeted, combined with a moderate
    exploration bonus.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division issues
    sigma = np.clip(sigma, 1e-10, None)
    
    # Standardized improvement: how many sigmas below f_best
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF, phi is standard normal PDF
    # This is the expected reduction in the best value if we evaluate x
    
    # Use efficient computation
    # For large positive z (mu << f_best), EI ~ f_best - mu (certain improvement)
    # For large negative z (mu >> f_best), EI ~ 0 (unlikely improvement)
    
    # Compute EI using the standard formula
    # To avoid numerical issues, use the stable form
    from scipy.special import erf
    from scipy.special import expit
    
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-0.5 * z^2) / sqrt(2*pi)
    
    # For numerical stability, handle different ranges of z
    # When z is very large positive, EI ≈ f_best - mu
    # When z is very negative, EI ≈ 0
    
    # Use a stable computation
    # EI = sigma * (z * Phi(z) + phi(z))
    # This form is more stable
    
    # Compute Phi(z) and phi(z)
    # Use the approximation for large |z|
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # EI = sigma * (z * Phi(z) + phi(z))
    EI = sigma * (z * Phi + phi)
    
    # Clip EI to be non-negative (numerical safety)
    EI = np.maximum(EI, 0.0)
    
    # Exploration bonus: UCB-style term
    # Use a moderate beta to balance exploration
    beta = 0.8
    UCB = beta * sigma
    
    # Combine: weighted sum of EI and UCB
    # EI is in objective units, UCB is also in objective units
    # Weight EI more heavily for exploitation
    score = (1.0 * EI + UCB) / (cost + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
