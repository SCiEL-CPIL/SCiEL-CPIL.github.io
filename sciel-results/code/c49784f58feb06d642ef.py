"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition function for Bayesian optimization.
    Uses a dynamic exploration coefficient that decreases as f_best improves,
    normalized by evaluation cost.
    """
    start_time = time.time()
    DEADLINE = 310.0  # Safety margin before 330s kill

    n = mu.shape[0]

    # Clamp sigma for numerical stability
    sigma_safe = np.clip(sigma, 1e-12, None)

    # Dynamic exploration coefficient:
    # Start with high exploration when f_best is far from minimum (0),
    # decrease as we get closer to the minimum.
    # f_best ranges from high values down to 0.
    # Use a schedule that keeps exploration meaningful early but exploits later.
    
    # Normalize f_best to get a progress indicator
    # The three-hump camel minimum is 0, maximum on [-5,5]^2 is around ~100+
    # Use a smooth decay: beta starts high and decreases
    f_best_safe = max(f_best, 1e-10)
    
    # Beta: exploration coefficient
    # When f_best is large (early), beta is large (more exploration)
    # When f_best is small (late), beta is smaller (more exploitation)
    # Scale: f_best ~ 50 initially, ~0 at minimum
    # beta = base + scale / (1 + f_best)
    base_beta = 1.0
    scale_beta = 3.0
    beta = base_beta + scale_beta / (1.0 + f_best_safe)
    
    # Clip beta to reasonable range
    beta = np.clip(beta, 0.5, 6.0)
    
    # LCB: mu - beta * sigma (lower is better for minimization)
    lcb = mu - beta * sigma_safe
    
    # Cost normalization: divide by cost to prioritize cheap evaluations
    cost_safe = np.maximum(cost, 1e-10)
    
    # Score: higher is better, so negate the cost-normalized LCB
    # -lcb / cost = (beta * sigma - mu) / cost
    scores = -lcb / cost_safe
    
    # Add a small exploration bonus for high-uncertainty, low-cost regions
    # This helps the GP model converge faster
    exploration_bonus = 0.15 * sigma_safe / cost_safe
    scores += exploration_bonus
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return simple UCB-like score as fallback
        simple = (beta * sigma_safe - mu) / cost_safe
        return np.where(np.isfinite(simple), simple, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
