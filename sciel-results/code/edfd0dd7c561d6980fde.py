"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erf


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing shekel 4d.
    
    Uses a Lower Confidence Bound (LCB) with a cost-scaled exploration bonus.
    The LCB explicitly trades off exploitation (low mu) against exploration (high sigma),
    and the exploration bonus is inversely scaled by cost to favor cheap, uncertain regions.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions (in objective units).
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    
    # Safety deadline guard: leave margin before 330s wall-clock
    DEADLINE = 320.0  # seconds
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Lower Confidence Bound (LCB) for minimization ---
    # LCB(x) = mu(x) - kappa * sigma(x)
    # We want to minimize LCB, so we maximize -LCB(x) = -(mu - kappa * sigma)
    # 
    # Key structural choice: the exploration bonus kappa is dynamically scaled
    # by the inverse of cost, so uncertain regions with low cost are prioritized.
    # This creates a cost-aware exploration signal that naturally balances
    # exploitation vs. exploration based on the cost landscape.
    
    # Base exploration parameter
    kappa_base = 1.0
    
    # Dynamic scaling: exploration bonus is inversely proportional to cost.
    # This means:
    # - Cheap points (low cost) get a larger effective exploration bonus
    # - Expensive points (high cost) get a smaller effective exploration bonus
    # This naturally encourages exploring cheap, uncertain regions.
    
    # Normalize cost to [0, 1] range for scaling
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min if cost_max > cost_min else 1.0
    cost_norm = (cost_safe - cost_min) / cost_range  # 0 = cheapest, 1 = most expensive
    
    # Inverse cost scaling: cheap points get higher multiplier
    # Use a power to control the aggressiveness of the cost penalty
    cost_multiplier = 1.0 / (1.0 + cost_norm)  # ranges from 1.0 (cheapest) to 0.5 (most expensive)
    
    # Effective kappa for each point
    kappa = kappa_base * cost_multiplier
    
    # Compute LCB: mu - kappa * sigma
    lcb = mu - kappa * sigma_safe
    
    # Acquisition score: maximize -LCB (i.e., minimize LCB)
    # The score is the negative of the LCB, so lower LCB = higher score
    scores = -lcb
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
