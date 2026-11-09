"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Hypar (EHypR) acquisition function.
    
    Maximizes E[max(f_best, f(x)) / f(x)] where f(x) ~ N(mu, sigma^2).
    
    This is equivalent to minimizing E[min(1, f(x)/f_best)].
    Let S = E[min(1, Y/f_best)].
    We return 1/S (with safeguards) to maximize the acquisition score.
    """
    if f_best <= 1e-8:
        # Fallback if we've essentially found the minimum
        return -mu

    c = f_best
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z_c = (c - mu) / sigma
    z_c = (c - mu) / sigma_safe
    
    # Phi(z_c) and phi(z_c)
    Phi_z = norm.cdf(z_c)
    phi_z = norm.pdf(z_c)
    
    # S = 1 - Phi_z + (mu/c)*Phi_z - (sigma/c)*phi_z
    # S = 1 + Phi_z*(mu/c - 1) - (sigma/c)*phi_z
    term1 = 1.0
    term2 = Phi_z * (mu / c - 1.0)
    term3 = (sigma_safe / c) * phi_z
    
    S = term1 + term2 - term3
    
    # The acquisition value is 1 / S
    # If S is negative, 1/S is negative (bad score).
    # If S is small positive, 1/S is large (good score).
    
    # Avoid division by zero
    # If S is very close to 0, clip it to a small value to prevent inf
    # However, we want large scores for small S.
    # So we only clip if it's exactly 0 or causing overflow.
    # Using nan_to_num on the result is safer.
    
    acq = 1.0 / S
    acq = np.nan_to_num(acq, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return acq
# EVOLVE-BLOCK-END
