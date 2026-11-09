import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Structural Change: Switch from Expected Improvement (EI) to UCB.
    - UCB balances exploitation (-mu) and exploration (sigma) linearly.
    - Cost-awareness is applied by dividing the entire UCB score by the evaluation cost.
    - This encourages the search to find low function values in cheap regions,
      while still exploring uncertain regions if they are sufficiently cheap.
    """
    # Safety deadline guard
    t_start = time.time()
    
    # Ensure sigma is positive
    sigma_safe = np.clip(sigma, 1e-8, None)
    cost_safe = np.clip(cost, 1.0, None)
    
    # UCB Score for minimization:
    # We want to minimize f(x).
    # Standard UCB for maximization is mu + beta * sigma.
    # For minimization, we can maximize -mu + beta * sigma.
    # Or minimize mu - beta * sigma.
    # Let's maximize the score.
    # Score = -mu + beta * sigma.
    
    # Adaptive beta?
    # A fixed beta is simpler and more robust.
    # Let's try beta = 2.0.
    beta = 2.0
    
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-aware: divide by cost
    # This favors cheap points.
    score = ucb_score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final deadline check
    if time.time() - t_start > 325:
        pass
        
    return score
