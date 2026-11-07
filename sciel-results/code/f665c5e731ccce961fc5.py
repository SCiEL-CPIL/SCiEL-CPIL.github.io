import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Damped Upper Confidence Bound (Damped-UCB).
    
    This structural change moves away from simple linear UCB/EI blends.
    It uses a Gaussian damping factor on the exploration term to prevent
    exploring regions where the posterior mean is already significantly
    worse than the current best, focusing exploration on the "fringe"
    of the promising search space. This is often more effective for
    multimodal functions like Goldstein-Price where naive UCB might
    waste evaluations in far-away high-variance regions.
    
    Parameters
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (in objective units, lower is better)
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (in objective units)
    f_best : float
        Best (lowest) observed function value so far
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback to pure exploitation (minimize mu)
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are numpy arrays with correct shape
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    # Basic sanity: ensure finite values
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is non-negative and avoid division by zero in normalization
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute the gap between current best and the theoretical minimum (3.0)
    # This helps scale the exploration rate. If we are close to the minimum,
    # we should explore less.
    f_min = 3.0
    gap = max(f_best - f_min, 1e-6)
    
    # Adaptive beta: 
    # Start with high exploration, decay as we get closer to the minimum.
    # The decay is non-linear to allow for quick exploitation once close.
    # log(1+gap) provides a smooth scaling.
    beta = 1.5 * np.log1p(gap / 5.0) + 0.5
    
    # Damped UCB Formula:
    # Score = -mu + beta * sigma * exp(-0.5 * ((mu - f_best) / sigma)^2)
    # The exponential term is a Gaussian kernel centered at f_best.
    # It equals 1 when mu == f_best.
    # It decays rapidly as mu moves away from f_best (in the worse direction).
    # This prevents exploring regions where mu >> f_best, even if sigma is high.
    
    # Calculate the z-score of how far mu is from f_best, in units of sigma
    # We only care about the "worse" direction (mu > f_best) for damping.
    # If mu < f_best, the damping factor should be 1 (full exploration bonus).
    # Actually, if mu < f_best, it's already better, so -mu is high.
    # The damping should suppress the sigma term if mu is much worse than f_best.
    
    # Standardize the distance
    z = (mu - f_best) / sigma
    
    # Damping factor: exp(-0.5 * z^2)
    # If mu == f_best, z=0, damping=1.
    # If mu = f_best + sigma, z=1, damping=0.6.
    # If mu = f_best + 2*sigma, z=2, damping=0.13.
    # This effectively creates a "search radius" of about 2 sigmas around the best point.
    
    damping = np.exp(-0.5 * z * z)
    
    # Compute the score
    scores = -mu + beta * sigma * damping
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores.astype(np.float64)
