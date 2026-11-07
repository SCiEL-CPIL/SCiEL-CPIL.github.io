import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and Upper Confidence Bound (UCB) with adaptive weighting.
    
    The score combines exploitation (EI) and exploration (UCB) to balance
    finding better values versus exploring uncertain regions.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Clamp cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute Expected Improvement for minimization
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Compute Phi(z) and phi(z)
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # EI formula
    EI = delta * Phi + sigma_safe * phi
    EI = np.maximum(EI, 0.0)
    
    # Compute UCB: mu - beta * sigma (for minimization, we want low mu and high sigma)
    # Use an adaptive beta based on the scale of the problem
    # Estimate scale from the range of mu
    mu_range = np.ptp(mu)
    if mu_range < 1e-10:
        mu_range = 1.0
    
    # Adaptive beta: larger when sigma is large relative to mu range
    beta = 0.5 + 0.5 * (np.median(sigma_safe) / (mu_range + 1e-10))
    beta = np.clip(beta, 0.1, 2.0)
    
    UCB = mu - beta * sigma_safe
    
    # Combine EI and UCB
    # Normalize both to have similar scale
    EI_scale = np.std(EI) + 1e-10
    UCB_scale = np.std(UCB) + 1e-10
    
    EI_norm = EI / EI_scale
    UCB_norm = -UCB / UCB_scale  # Negate because lower UCB is better
    
    # Weighted combination: 60% EI, 40% UCB
    combined = 0.6 * EI_norm + 0.4 * UCB_norm
    
    # Cost-normalized score
    scores = combined / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if this took too long, return a simpler heuristic
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Fallback: simple expected improvement per cost
        simple_ei = np.maximum(f_best - mu, 0.0) + 0.5 * sigma_safe
        simple_scores = simple_ei / cost_safe
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores
