import numpy as np
from scipy.stats import norm
import time

_state = {
    "start_time": None
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    This is a structural change from UCB, focusing on the expected magnitude of improvement
    over the current best, which often performs better in multimodal landscapes by
    naturally balancing exploration and exploitation based on the posterior distribution.
    """
    if _state["start_time"] is None:
        _state["start_time"] = time.time()
    
    elapsed = time.time() - _state["start_time"]
    if elapsed > 325.0:
        # Fallback to exploitation if near deadline
        mu = np.asarray(mu, dtype=np.float64)
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # Calculate Z-score for Expected Improvement
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Calculate CDF and PDF
    # Use scipy.stats.norm for accuracy
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # Expected Improvement for minimization:
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
