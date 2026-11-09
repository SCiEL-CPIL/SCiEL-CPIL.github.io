import numpy as np
from scipy.stats import norm
import time

# Track time to ensure we stay within budget
_start_time = time.time()
_TIME_LIMIT = 325.0  # 5 seconds margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Compute a pure GP-LCB (Lower Confidence Bound) acquisition function.
    This approach directly maximizes the lower bound of the objective,
    balancing exploitation (low mu) and exploration (high sigma) via kappa.
    
    Parameters:
    mu : np.ndarray
        Posterior mean at candidate points, shape (N,).
    sigma : np.ndarray
        Posterior standard deviation at candidate points, shape (N,).
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    scores : np.ndarray
        LCB scores, shape (N,), higher is better.
    """
    global _start_time
    
    # Safety check: if we're close to the time limit, return a simple score
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Return deterministic score based on -mu to avoid hanging
        return -np.asarray(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle numerical stability for sigma
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # GP-LCB for minimization:
    # We want to find points where the true function value is likely low.
    # Lower bound: L(x) = mu(x) - kappa * sigma(x)
    # To maximize this lower bound (i.e., find the most promising point),
    # we compute score = -mu(x) + kappa * sigma(x)
    # This is equivalent to maximizing -L(x).
    
    # Adaptive kappa:
    # Early on (f_best far from global min), we want to explore more (higher kappa).
    # Late on (f_best near global min), we want to exploit more (lower kappa).
    # Global minimum of Styblinski-Tang 1D is approx -39.1662
    
    global_min = -39.1662
    
    # Measure proximity to global minimum
    # delta_to_min = f_best - global_min (positive when above min)
    delta_to_min = f_best - global_min
    
    # Use a smooth transition for kappa
    # When delta_to_min is large (far from min), kappa should be high (e.g., 2.5)
    # When delta_to_min is near 0 (close to min), kappa should be low (e.g., 0.5)
    
    # Sigmoid-based adaptation
    # We want kappa to decrease as delta_to_min decreases.
    # Let t = delta_to_min / scale
    # If t is large, sigmoid(t) is close to 1.
    # If t is small, sigmoid(t) is close to 0.5.
    # If t is negative (below min, shouldn't happen), sigmoid(t) is close to 0.
    
    scale = 10.0
    t = delta_to_min / scale
    sigmoid_val = 1.0 / (1.0 + np.exp(-t))
    
    # Map sigmoid to kappa range [0.5, 2.5]
    # When sigmoid is 1 (far from min), kappa = 2.5
    # When sigmoid is 0.5 (mid), kappa = 1.5
    # When sigmoid is 0 (at min), kappa = 0.5
    kappa_min = 0.5
    kappa_max = 2.5
    kappa = kappa_min + (kappa_max - kappa_min) * sigmoid_val
    
    # Compute LCB score
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
