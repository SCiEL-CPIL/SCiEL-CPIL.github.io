# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Lower Confidence Bound (LCB) acquisition function.
    
    Uses score = mu + kappa * sigma - lambda * cost to directly target
    the global minimum in cheap regions, avoiding the local improvement
    bias of EI-based methods.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from GP.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from GP.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margins
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute a cost-aware exploration coefficient.
    # Normalize cost to [0,1] range for scaling kappa.
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_norm = np.zeros_like(cost_safe)
    else:
        cost_norm = (cost_safe - cost_min) / cost_range
    
    # Base LCB: mu + kappa * sigma
    # Use a moderate kappa that balances exploration and exploitation.
    # The cost penalty term - lambda * cost encourages cheap evaluations.
    
    # Adaptive kappa: slightly larger when uncertainty is high relative to 
    # the spread of mu, to encourage exploration in uncertain regions.
    mu_spread = np.ptp(mu)
    if mu_spread < 1e-9:
        mu_spread = 1e-9
    sigma_mean = np.mean(sigma_safe)
    
    # Scale kappa based on relative uncertainty
    relative_uncertainty = sigma_mean / mu_spread
    kappa = 1.0 + 0.5 * relative_uncertainty
    
    # Cost penalty coefficient: penalize expensive evaluations
    # Scale lambda so that the cost penalty is meaningful relative to 
    # the typical improvement scale
    lambda_cost = 0.1 * (f_best - np.min(mu)) / np.mean(cost_safe) if f_best > np.min(mu) else 0.1
    
    # Compute LCB scores (we want to minimize, so higher score = better)
    # LCB = mu - kappa * sigma (for minimization)
    # But we want higher scores to be selected, so we negate:
    # score = -(mu - kappa * sigma) + bonus - lambda * cost
    # Which simplifies to: -mu + kappa * sigma - lambda * cost
    
    # Actually, for minimization, LCB = mu - kappa*sigma
    # We want to select points with LOW LCB, so score = -LCB = -mu + kappa*sigma
    # Then subtract cost penalty: score = -mu + kappa*sigma - lambda*cost
    
    scores = -mu + kappa * sigma_safe - lambda_cost * cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
