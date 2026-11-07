"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-adjusted Lower Confidence Bound (LCB) acquisition function.
    
    The exploitation term (-mu) is kept invariant, while the exploration term
    (sigma) is scaled by the inverse cost (1/cost). This creates a non-uniform
    exploration landscape that prioritizes high-uncertainty, low-cost areas,
    while still allowing exploitation of low-mean regions regardless of cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-10, None)
    
    # Cost-adjusted LCB:
    # The exploitation signal (-mu) is preserved as-is.
    # The exploration bonus (sigma) is penalized by cost: sigma / cost.
    # This means high-cost regions get a smaller exploration bonus,
    # while low-cost regions get a larger bonus for the same uncertainty.
    
    # Use a beta parameter to balance exploration vs exploitation.
    # A larger beta means more exploration.
    # We use a moderate value that adapts to the cost structure.
    beta = 2.0
    
    # LCB score: higher is better
    # score = -mu + beta * (sigma / cost)
    scores = -mu + beta * (sigma / cost)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
