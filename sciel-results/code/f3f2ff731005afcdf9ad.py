import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition function for Bayesian optimization (styblinski tang 2d).
    
    Uses a pure cost-normalized Lower Confidence Bound:
    score(x) = -(mu(x) - kappa * sigma(x)) / cost(x)
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Deadline check
    def time_left():
        return 330.0 - (time.time() - t0) - 5.0
    
    if time_left() <= 0:
        # Fallback: simple LCB
        kappa = 2.0
        scores = (-mu + kappa * sigma) / cost
        return np.where(np.isfinite(scores), scores, -1e10)
    
    # Structural Change: Cost-Normalized Expected Improvement (EI)
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    
    z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values causing numerical issues in norm.cdf/pdf
    # norm.cdf handles large values, but let's be safe
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate EI
    # Note: If sigma is very small, z is large.
    # If mu < f_best, z is large positive, Phi(z) ~ 1, phi(z) ~ 0. EI ~ f_best - mu.
    # If mu > f_best, z is large negative, Phi(z) ~ 0, phi(z) ~ 0. EI ~ 0.
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Cost-normalized score
    scores = ei / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
