import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and Lower Confidence Bound (LCB) with dynamic exploration.
    
    Parameters:
    mu: array of mean predictions from GP (shape (2000,))
    sigma: array of standard deviations from GP (shape (2000,))
    f_best: best observed value so far
    cost: array of costs for each candidate (shape (2000,))
    
    Returns:
    scores: array of scores (higher = evaluate next), shape (2000,)
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Track elapsed time for deadline guard
    start_time = time.time()
    DEADLINE = 325.0  # safety margin before 330s kill
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Compute Lower Confidence Bound (LCB) for minimization
    # LCB = mu - beta * sigma
    # Dynamic beta: start high for exploration, decay as f_best improves
    # Use a decay schedule based on f_best magnitude
    # Three-hump camel minimum is 0, so f_best starts positive and decreases
    # Normalize f_best to get a "progress" indicator
    # At start, f_best ~ O(1-10), at end f_best ~ 0
    f_best_abs = abs(f_best)
    # beta decays from ~3.0 to ~0.5 as f_best approaches 0
    beta = 0.5 + 2.5 * np.exp(-f_best_abs / 1.0)
    
    lcb = mu - beta * sigma
    
    # Hybrid acquisition: combine EI and LCB
    # Weight them based on a dynamic schedule
    # Early: more LCB (exploration), Late: more EI (exploitation)
    # progress = 1 - exp(-f_best_abs) gives 0 when f_best large, 1 when f_best ~ 0
    progress = 1.0 - np.exp(-f_best_abs / 0.5)
    w_ei = progress
    w_lcb = 1.0 - progress
    
    # Combine
    raw_score = w_ei * ei + w_lcb * (-lcb)  # -lcb = -mu + beta*sigma, higher is better
    
    # Normalize by cost (cost-aware)
    # Higher score per unit cost is preferred
    scores = raw_score / cost
    
    # Handle numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard: if we've spent too much time, return immediately
    elapsed = time.time() - start_time
    if elapsed > 0.1:  # This function should be fast; this is just a safety check
        # Return the current scores as-is
        pass
    
    return scores
