# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for deadline guard
_start_time = None

def _init_state():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for 2000 candidates.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    global _start_time
    
    # Initialize state on first call
    _init_state()
    
    # Deadline guard
    # The total budget is 330s. We should return well before that.
    # This function is O(N) and very fast, but we check just in case.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Return a safe default if we are critically close to the deadline
        # This is unlikely to happen given the O(N) complexity, but required by directive.
        return np.zeros(len(mu))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety clamps
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Cost-Adjusted Expected Improvement (CA-EI)
    # Standard EI for minimization:
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    # where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    # To handle numerical stability and vectorization:
    z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values that might cause issues, though norm functions handle inf well
    # However, for very large positive z (mu << f_best), EI is roughly f_best - mu.
    # For very large negative z (mu >> f_best), EI is roughly 0.
    
    # We compute EI only where sigma is significant, but the formula works generally.
    # Using scipy.stats.norm for precision.
    
    # To optimize speed, we can use the approximation or vectorized scipy calls.
    # scipy.stats.norm.pdf and cdf are vectorized.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: (f_best - mu) = z * sigma, so EI = sigma * (z * Phi(z) + phi(z))
    # This form is numerically more stable and avoids large multiplications.
    
    ei = sigma * (z * Phi_z + phi_z)
    
    # Ensure EI is non-negative (due to floating point errors, it might be slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Adjusted Score: EI / cost
    # This maximizes expected improvement per unit cost.
    scores = ei / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
