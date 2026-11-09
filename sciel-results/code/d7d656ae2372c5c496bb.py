import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START
def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Pure Expected Improvement (EI).
    
    This function calculates the Expected Improvement for each candidate point.
    EI is a probabilistic acquisition function that balances exploration and
    exploitation by estimating the expected amount by which a candidate point
    might improve upon the current best observed value.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions at candidate points.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means more promising.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 330s limit, leave 40s margin
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is non-negative. Higher is better.
    scores = ei
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
