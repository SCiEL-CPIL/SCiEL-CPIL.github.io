# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 325.0  # Safety margin vs 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses Expected Improvement (EI) for minimization.
    EI balances exploitation (lowering the predicted mean) and exploration
    (reducing uncertainty) by estimating the expected reduction in the
    objective value relative to the current best.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Deadline check
    if time.time() > _DEADLINE - 1.0:
        return np.zeros(len(mu))
    
    # Ensure sigma is finite and non-negative, with a small floor to prevent division by zero
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-8)
    
    fb = float(f_best)
    
    # Calculate the standardized improvement Z
    # Z = (f_best - mu) / sigma
    z = (fb - mu) / sigma
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Use scipy.stats.norm for numerical stability
    phi_z = norm.pdf(z)
    phi_z_cdf = norm.cdf(z)
    
    ei = (fb - mu) * phi_z_cdf + sigma * phi_z
    
    # Ensure output is finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-START
