"""Cost-aware Thompson Sampling acquisition for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition: draw one posterior sample at each candidate
    and score by (sampled_value / cost). Lower sampled value and lower cost
    both increase the score (we return negative of ratio so higher = better).
    
    Actually, we want to MINIMIZE the function, so we want low mu and low sigma
    (uncertainty encourages exploration). Thompson Sampling: sample y_i ~ N(mu_i, sigma_i^2),
    then pick the candidate with minimum y_i / cost_i.
    
    To convert to a "higher is better" score: score = -y_sample / cost
    """
    t0 = time.time()
    deadline = t0 + 295.0  # safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety
    safe_cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-8)
    
    # Draw a single standard normal sample for each candidate
    # Use a fixed seed per call? No - Thompson Sampling should be stochastic.
    # But we need reproducibility? The task says "each evaluation must finish within 300s"
    # and the function is called repeatedly in a loop. Let's use random samples.
    
    # For numerical stability, clip sigma
    z = np.random.standard_normal(n)
    sampled = mu + sigma * z
    
    # Score: we want to minimize sampled / cost, so higher score = lower ratio
    # score = -sampled / cost
    scores = -sampled / safe_cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    elapsed = time.time() - t0
    if elapsed > deadline - t0:
        # Shouldn't happen for 2000 points, but guard anyway
        pass
    
    return scores
# EVOLVE-BLOCK-END
