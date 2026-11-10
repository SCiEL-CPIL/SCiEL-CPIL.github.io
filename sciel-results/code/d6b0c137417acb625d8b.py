"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s hard kill
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # Structural Change: Cost-Discounted Probability of Improvement (CD-PI)
    # Strategy: Shift from maximizing expected magnitude of improvement (EI) 
    # to maximizing the frequency of successful updates (POI) per unit cost.
    # This is more robust to GP uncertainty and cost variations.
    
    # 1. Compute Probability of Improvement (POI)
    # POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    poi = norm.cdf(z)
    
    # 2. Cost Discounting
    # The directive suggests normalizing by the square root of the evaluation cost.
    # This penalizes expensive points less severely than linear cost (EI/cost) 
    # but more than no cost penalty. It encourages finding cheap improvements 
    # without completely ignoring expensive high-probability regions.
    cost_discount = np.sqrt(cost)
    
    # 3. Final Score
    scores = poi / cost_discount
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: just return the scores computed so far (which are already vectorized and fast)
        pass
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
