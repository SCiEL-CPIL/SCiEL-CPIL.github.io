import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start = time.time()
    limit = start + 5.0
    
    # Upper Confidence Bound (UCB)
    # Structural change from EI: Linear balance of mean and std vs Non-linear EI.
    # Higher score means better (lower objective).
    # We minimize f, so we maximize -mu + beta * sigma.
    
    # Adaptive beta: Explore more when f_best is far from global min (0.397887)
    # Scale f_best relative to typical Branin range
    # If f_best is large, beta is larger.
    # f_best range ~ 0.4 to 300.
    # Let's use a simple heuristic.
    beta = 2.0
    
    scores = -mu + beta * sigma
    
    # Ensure finite
    if not np.all(np.isfinite(scores)):
        # Fallback to EI
        from scipy.stats import norm
        z = (f_best - mu) / (sigma + 1e-9)
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
    # Check time (though it's instant)
    if time.time() > limit:
         # If we were doing something complex, we'd return partial. 
         # Here we just return what we have.
         pass
         
    return scores
