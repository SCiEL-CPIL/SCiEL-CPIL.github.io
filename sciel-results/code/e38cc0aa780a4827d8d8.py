"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI).
    This strategy favors points with a high probability of improving upon f_best,
    which inherently balances exploitation (low mu) and exploration (high sigma).
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, np.nanmax(mu) if np.any(np.isfinite(mu)) else 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    sigma = np.maximum(sigma, 1e-12)
    
    # Calculate Z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # PI = P(f(x) < f_best) = Phi(z)
    z = (f_best - mu) / sigma
    
    # Compute PI
    scores = norm.cdf(z)
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
