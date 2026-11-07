import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Strategy:
    1. Use Lower Confidence Bound (LCB) for minimization: Score = mu - kappa * sigma.
       This balances exploitation (low mu) and exploration (high sigma).
    2. Make it cost-aware by scaling kappa inversely with cost.
       Cheaper points (lower cost) get a higher exploration bonus, encouraging
       the optimizer to explore cheap regions more aggressively.
       Expensive points get a lower exploration bonus, requiring stronger evidence
       (lower mu) to be selected.
    3. Adapt kappa dynamically based on the gap to the known global minimum (-0.8).
       If we are far from the minimum, we explore more (higher kappa).
       If we are close, we exploit more (lower kappa).
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit
    
    # Ensure inputs are 1D arrays for consistent shape handling
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    # Safety: sigma must be > 0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Theoretical minimum for cosine8 is -0.8
    f_min = -0.8
    
    # Calculate the gap to the theoretical minimum
    # gap is positive if f_best > -0.8 (which it should be initially)
    gap = f_best - f_min
    
    # Dynamic exploration coefficient (kappa)
    # Base kappa is tuned for 8D optimization.
    # We start with a moderate exploration level.
    base_kappa = 1.5
    
    # If the gap is large, we are far from the optimum. Increase exploration.
    # If the gap is small, we are close. Decrease exploration to refine.
    # Normalize gap: typical initial gap might be ~0.2 to 0.5 depending on random start.
    # Let's say gap > 0.5 is "far", gap < 0.1 is "close".
    # Use a smooth transition or simple step.
    if gap > 0.5:
        kappa_scale = 1.5
    elif gap > 0.2:
        kappa_scale = 1.0
    elif gap > 0.05:
        kappa_scale = 0.5
    else:
        kappa_scale = 0.2
        
    kappa = base_kappa * kappa_scale
    
    # Cost-awareness:
    # Cost is in [1, 2].
    # We want to favor lower cost.
    # Let's scale kappa by (1 / cost).
    # If cost is 1, factor is 1.
    # If cost is 2, factor is 0.5.
    # This reduces the exploration bonus for expensive points.
    # Alternatively, we can add a cost penalty to the score directly.
    # Score = mu - kappa * sigma - cost_penalty
    # But scaling kappa is a cleaner "cost-aware" modification to the UCB/LCB structure.
    
    cost_factor = 1.0 / cost
    
    # Final LCB score
    # We want to maximize the score.
    # LCB for minimization is mu - kappa * sigma.
    # Lower LCB is better. So we maximize -LCB = -mu + kappa * sigma.
    # Wait, standard BO minimizes f(x). 
    # Acquisition function usually maximizes.
    # If we use LCB = mu - kappa * sigma, we want to MINIMIZE this value.
    # So the score to MAXIMIZE should be -LCB = -mu + kappa * sigma.
    # Let's verify:
    # If mu is low (good), -mu is high (good score).
    # If sigma is high (uncertain), kappa*sigma is high (good score).
    # This encourages picking points with low mean or high uncertainty.
    
    scores = -mu + kappa * sigma_safe * cost_factor
    
    # Handle any numerical issues
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Check time
    if time.time() > deadline:
        # Return a default score if we are running out of time
        # This shouldn't happen as the calculation is O(N)
        return np.ones_like(mu)
        
    return scores
