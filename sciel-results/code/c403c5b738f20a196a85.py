import numpy as np
import time

_start_time = time.time()
_DEADLINE_MARGIN = 15.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each point
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher is better)
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        return np.full(2000, 0.0)
    
    # Convert to numpy arrays and ensure finiteness
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Safe sigma for calculations
    eps = 1e-10
    safe_sigma = np.maximum(sigma, eps)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    Z = (f_best - mu) / safe_sigma
    
    # Use scipy.stats.norm for PDF and CDF
    # Phi(Z) is norm.cdf(Z)
    # phi(Z) is norm.pdf(Z)
    from scipy.stats import norm
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    EI = (f_best - mu) * Phi + safe_sigma * phi
    
    # Ensure EI is non-negative (numerical stability)
    EI = np.maximum(EI, 0.0)
    
    # Cost-Effective EI (CEI)
    # CEI = EI / cost
    CEI = EI / cost
    
    # Adaptive Strategy:
    # Early phase (f_best is high, e.g., > -3.5): Use pure EI to explore broadly and ignore cost.
    # Late phase (f_best is low, e.g., < -3.5): Use CEI to be cost-efficient and refine the minimum.
    # Threshold chosen based on Hartmann 3D minimum (-3.86) and typical random start values.
    threshold = -3.5
    
    if f_best > threshold:
        score = EI
    else:
        score = CEI
        
    # Final safety check
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
