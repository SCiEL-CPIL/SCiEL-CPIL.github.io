import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: Array of shape (2000,) with posterior means.
        sigma: Array of shape (2000,) with posterior standard deviations.
        f_best: The lowest observed function value.
        
    Returns:
        Array of shape (2000,) with scores. Higher scores are preferred.
    """
    # Start timer for safety guard
    start_time = time.time()
    
    # UCB parameter (beta)
    # In 2D with a small number of evaluations (30), aggressive exploration is often beneficial.
    # beta=1.0 is a standard choice that balances exploitation (-mu) and exploration (sigma).
    beta = 1.0
    
    # Calculate UCB scores: Maximize -mu + beta * sigma
    # Since we are minimizing the objective, we want low mu and high sigma.
    # The score should be high for points that are likely to be better (low mu) or uncertain (high sigma).
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Time guard: If computation took too long (unlikely for vectorized ops), 
    # we still return the computed scores as they are ready.
    # The guard is primarily to satisfy the directive structure.
    elapsed = time.time() - start_time
    if elapsed > 290:
        # In a real scenario, we might truncate or simplify, but here 
        # the operation is instantaneous.
        pass
        
    return scores
