"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds, evaluator kills at 330s

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining UCB and Expected Improvement for
    multimodal optimization (Michalewicz 2D).
    
    Uses a higher UCB beta to aggressively explore unvisited regions,
    combined with EI for exploitation. No normalization is applied so
    that the natural scales of UCB and EI are preserved.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (lowest, since we're minimizing).
    
    Returns:
    --------
    scores : np.ndarray, shape (n,)
        Acquisition scores; higher means more desirable to evaluate.
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        return -np.asarray(mu, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # UCB for minimization: higher score = more promising
    # Use a larger beta to strongly favor exploration in early iterations
    beta = 4.0
    ucb = -mu + beta * sigma
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Combine: UCB dominates for exploration, EI provides exploitation signal
    # Scale EI to be comparable to UCB by using the typical magnitude of sigma
    # This avoids the normalization that destroys scale information
    sigma_ref = np.median(sigma)
    if sigma_ref < 1e-9:
        sigma_ref = 1e-9
    ei_weighted = ei / (sigma_ref + 1e-9)
    
    scores = ucb + ei_weighted
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
