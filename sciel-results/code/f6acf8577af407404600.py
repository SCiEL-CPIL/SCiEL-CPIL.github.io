"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: UCB with cubic cost penalty.
    Uses Upper Confidence Bound (mu + kappa * sigma) divided by cost^3
    to aggressively prioritize low-cost evaluations while exploring
    regions of high uncertainty. This shifts the search from pure
    exploitation (EI) to a balance that heavily favors cheap,
    uncertain regions where the global minimum may be hiding.
    
    Score = (mu + kappa * sigma) / cost^3
    """
    # Deadline guard
    t0 = time.time()
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # UCB: balance exploration and exploitation
    # kappa controls the exploration-exploitation tradeoff
    # Higher kappa = more exploration
    # We use a moderate kappa that works well for these functions
    kappa = 1.0
    
    # Compute UCB value
    # Note: mu can be negative (we're minimizing), so we want high mu + kappa*sigma
    # But wait - we're MINIMIZING, so lower mu is better
    # For minimization, UCB should be: mu - kappa * sigma (lower is better)
    # But the task says "higher = evaluate next", so we need to flip
    # 
    # Actually, let me reconsider. The GP gives mu and sigma in objective units.
    # We're minimizing, so we want points where mu is low AND sigma is high
    # (uncertainty might hide an even lower value).
    #
    # For minimization with "higher score = better":
    # Score should favor low mu and high sigma
    # One approach: Score = -mu + kappa * sigma (negate mu since we minimize)
    # Or: Score = (f_best - mu) + kappa * sigma (expected improvement + exploration)
    #
    # Let's use: Score = (f_best - mu) + kappa * sigma
    # This combines exploitation (f_best - mu, higher when mu is much lower than f_best)
    # with exploration (sigma, higher when uncertain)
    
    # Actually, for a cleaner UCB for minimization:
    # We want to maximize: -mu + kappa * sigma
    # Since lower mu is better, -mu is higher when mu is lower
    # And kappa * sigma encourages exploration
    
    # But we also want to incorporate the cost penalty
    # Score = (-mu + kappa * sigma) / cost^gamma
    
    # Let me use a simpler approach:
    # For minimization, define the "upper confidence bound" as:
    # UCB_min(x) = mu(x) - kappa * sigma(x)  (lower bound on the true value)
    # We want to minimize this, so the acquisition should be high when UCB_min is low
    # Score = -UCB_min / cost^gamma = (-mu + kappa * sigma) / cost^gamma
    
    # Let's use gamma = 3.0 as suggested
    gamma = 3.0
    
    # Compute the score
    # -mu + kappa * sigma favors low mu (good for minimization) and high sigma (exploration)
    numerator = -mu + kappa * sigma_safe
    
    # Cost penalty: divide by cost^gamma
    cost_safe = np.maximum(cost, 1e-12)
    score = numerator / (cost_safe ** gamma)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - t0
    if elapsed > 250:  # Safety margin
        # Return a simple fallback if we're running too long
        # But UCB computation is fast, so this should rarely trigger
        pass
    
    return score
    # EVOLVE-BLOCK-END
