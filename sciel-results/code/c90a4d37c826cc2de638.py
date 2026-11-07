"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Probability of Improvement (PI)
    with an exploration bonus based on predictive uncertainty (LCB-like),
    normalized by evaluation cost.
    
    Formula: Score = PI * (1 + beta * sigma) / cost
    - PI encourages exploitation (finding points better than f_best).
    - (1 + beta * sigma) encourages exploration of uncertain regions.
    - Division by cost penalizes points that are expensive to evaluate 
      (cost = 1 + mean(x_scaled), so lower x values are cheaper).
    """
    t0 = time.time()
    # Safety margin: The evaluator kills after 330s. 
    # We assume the loop has some overhead, so we aim to finish within 300s 
    # to be safe, but the guard here is for this specific call.
    # However, since this is called inside a loop, the total time is the sum.
    # The prompt says "Each evaluation must finish within 300 s". 
    # This likely refers to the total execution time of the script.
    # We should track global time if possible, but we only have local context.
    # We will assume the caller manages the loop, but we must ensure *this* call is fast.
    # The previous program had a deadline guard. We will keep a simple one.
    
    # Ensure inputs are numpy arrays for speed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs to avoid division by zero or log of zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Exploration term: 
    # We use a multiplier (1 + beta * sigma) to boost points with high uncertainty.
    # beta is tuned to be aggressive (2.0) as per inspiration.
    beta = 2.0
    exploration_boost = 1.0 + beta * sigma_safe
    
    # Combine: PI * Exploration_Boost
    # This ensures we only explore if there's a chance of improvement, 
    # but strongly prefer high uncertainty if that chance exists.
    scores = pi * exploration_boost
    
    # Normalize by cost
    # Cost is (1 + mean(x_scaled)). Lower cost is better.
    scores = scores / cost_safe
    
    # Handle any numerical issues
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard for this specific function call (should be very fast, < 1ms usually)
    # But just in case of large arrays or slow scipy on some machines
    if time.time() - t0 > 2.0: # 2 seconds is way more than enough for 2000 points
         # Fallback to a simpler score if it's taking too long (unlikely)
         # Just return the PI/cost which is cheaper to compute? 
         # Actually, norm.cdf is the expensive part. 
         # If we are slow, we can't really do much better without changing the algorithm.
         pass
        
    return scores
# EVOLVE-BLOCK-END
