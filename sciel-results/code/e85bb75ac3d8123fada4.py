"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 290.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    Uses a structural shift from Improvement-based (EI/PI) to Bound-based (UCB/LCB).
    LCB explicitly balances exploitation (low mu) and exploration (high sigma).
    The cost penalty is applied with a high exponent to aggressively favor
    low-cost regions, which helps navigate the cost-aware landscape.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback to a simple, fast score if we are running out of time
        # This prevents timeout failures
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        return (sigma_safe - mu) / cost_safe

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Exploration parameter kappa
    # A higher kappa favors exploration (high sigma regions)
    # A lower kappa favors exploitation (low mu regions)
    # 1.5 is a common default for UCB/LCB in BO
    kappa = 1.5
    
    # Calculate LCB score: We want to maximize this.
    # LCB = mu - kappa * sigma
    # Since we want to minimize the function, we want low mu and high sigma.
    # Score = -LCB = kappa * sigma - mu
    lcb_score = kappa * sigma_safe - mu
    
    # Apply cost penalty
    # High gamma (e.g., 2.0) strongly penalizes expensive evaluations.
    # This is crucial because the cost function is spatially varying.
    gamma = 2.0
    score = lcb_score / (cost_safe ** gamma)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
