# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import ndtr, ndtr_inv
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 320.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Strategy: Standard Expected Improvement (EI) with a robust implementation.
    We use scipy.special.ndtr for the CDF to ensure speed and numerical stability.
    The score is EI + a small constant to ensure all scores are positive, 
    which can help in some downstream selection mechanisms if they assume 
    non-negative scores, though typically relative ordering is what matters.
    However, the key difference from the current 'portfolio' approach is that 
    we do NOT dynamically switch acquisition functions based on a heuristic 
    that might pick a suboptimal variant (e.g., pure exploitation) early on. 
    EI is theoretically sound for minimizing simple regret and adapts naturally 
    to the balance of exploration and exploitation via the sigma term.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    fb = float(f_best)
    
    # Ensure sigma is strictly positive to avoid division by zero
    # The problem states sigma is a standard deviation, so it should be >= 0.
    # We clip to a small epsilon to be safe.
    eps = 1e-12
    s = np.maximum(sigma, eps)
    
    # Calculate Expected Improvement
    # EI(x) = (fb - mu) * Phi(z) + sigma * phi(z)
    # where z = (fb - mu) / sigma
    # Phi is the standard normal CDF
    # phi is the standard normal PDF
    
    z = (fb - mu) / s
    
    # Use scipy.special.ndtr for CDF (fast and accurate)
    cdf_z = ndtr(z)
    
    # Calculate PDF
    # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    pdf_z = norm.pdf(z)
    
    ei = (fb - mu) * cdf_z + s * pdf_z
    
    # EI is always non-negative.
    # To be absolutely safe against any "higher is better" logic that might 
    # get confused by zeros if all candidates are worse than f_best (unlikely 
    # in BO but possible), we return EI directly. 
    # The evaluator picks the candidate with the HIGHEST score.
    # EI is higher where improvement is more likely/expected.
    
    # Check for finite values
    # If any values are not finite, replace them with 0.0 (lowest possible EI)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
