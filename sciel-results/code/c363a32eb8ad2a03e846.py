# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Hartmann 3D function definition
def hartmann3d(x):
    """Evaluate Hartmann 3D function at points x (N, 3)."""
    alpha = np.array([1.0, 10.0, 30.0, 3.0, 36.0, 10.0, 30.0, 10.0, 10.0, 10.0,
                      30.0, 3.0, 36.0, 10.0, 10.0, 10.0, 30.0, 3.0, 36.0, 10.0])
    A = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0]
    ])
    P = 2 * np.pi * np.array([
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0]
    ])
    
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    result = np.zeros(x.shape[0])
    for j in range(20):
        term = alpha[j] * np.sum(A[j] * (x**2 - P[j] * x + P[j]**2 / 4.0), axis=1)
        result -= term
    
    return result


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a combination of Expected Improvement and
    a rank-based uncertainty bonus to encourage exploration.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : array_like, shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best observed value so far (lowest)
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores, higher means more promising
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is finite and non-negative
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Compute rank-based uncertainty bonus
    # Rank candidates by sigma (uncertainty) and give higher bonus to more uncertain points
    # But gate by promise: only boost uncertainty if the mean is reasonably promising
    # (i.e., not far worse than f_best)
    
    # Rank sigma values (higher sigma = higher rank index)
    sigma_ranks = np.argsort(np.argsort(sigma)).astype(np.float64) / (len(sigma) - 1)
    
    # Promise gate: how promising is the mean? 
    # Use a soft threshold: points with mu much worse than f_best get less uncertainty boost
    # Normalize (f_best - mu) relative to sigma to get a "promise" score
    promise_z = (f_best - mu) / sigma
    # Clip promise to avoid extreme values
    promise_z = np.clip(promise_z, -3, 3)
    # Convert to [0, 1] using sigmoid-like function
    promise_gate = 1.0 / (1.0 + np.exp(-promise_z))
    
    # Combine: uncertainty bonus weighted by promise
    # Higher uncertainty + higher promise = higher score
    uncertainty_bonus = sigma_ranks * promise_gate
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Weighted combination
    # 75% EI, 25% uncertainty bonus
    scores = 0.75 * ei_norm + 0.25 * uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
