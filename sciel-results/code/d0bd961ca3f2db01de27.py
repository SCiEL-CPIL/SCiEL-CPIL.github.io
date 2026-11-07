import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more desirable points.
    """
    start_time = time.time()
    deadline = start_time + 295  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure valid inputs
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Estimate number of evaluations so far from cost
    # Initial 5 points have cost in [1, 2], average ~1.5
    # We can estimate n_evals from the GP, but we don't have that info directly
    # Use a heuristic: assume we're in early-to-mid stage
    # Adaptive beta: start high for exploration, decay to lower for exploitation
    
    # Use a simple heuristic based on f_best and sigma scale
    # If f_best is still high, we need more exploration
    sigma_scale = np.median(sigma)
    
    # Adaptive beta: higher when sigma is large relative to f_best
    # This encourages exploration early on
    if f_best > 0:
        beta_base = 2.0 + 1.0 * np.exp(-f_best / (1.0 + sigma_scale))
    else:
        beta_base = 1.5
    
    # Cost-aware EI (Expected Improvement)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # For cost-awareness, divide by cost^alpha where alpha < 1 to soften cost penalty
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    # Use scipy if available, else approximate
    try:
        from scipy.stats import norm
        Phi = norm.cdf(z)
        phi = norm.pdf(z)
    except ImportError:
        # Approximation using erf
        Phi = 0.5 * (1 + np.erf(z / np.sqrt(2)))
        phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    
    ei = (f_best - mu) * Phi + sigma * phi
    ei = np.maximum(ei, 0)
    
    # Cost-aware: divide by cost^0.7 (softer than cost^1)
    cost_alpha = 0.7
    cost_aware_ei = ei / (cost ** cost_alpha)
    
    # Add UCB component for exploration
    ucb = beta_base * sigma / (cost ** 0.5)
    
    # Combine: weighted sum of cost-aware EI and UCB
    # Weight shifts from exploration to exploitation as f_best improves
    if f_best > 1.0:
        w_ei = 0.5
        w_ucb = 0.5
    elif f_best > 0.1:
        w_ei = 0.7
        w_ucb = 0.3
    else:
        w_ei = 0.85
        w_ucb = 0.15
    
    scores = w_ei * cost_aware_ei + w_ucb * ucb
    
    # Gentle penalty for very high cost points (cost > 1.8)
    high_cost_mask = cost > 1.8
    if np.any(high_cost_mask):
        scores[high_cost_mask] *= 0.85
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 290:
        # Return immediately if close to deadline
        return scores
    
    return scores
