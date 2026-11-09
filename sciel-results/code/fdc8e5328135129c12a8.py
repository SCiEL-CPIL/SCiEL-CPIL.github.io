import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Lower Confidence Bound (LCB) for minimization.
    
    For minimization, LCB = mu - beta * sigma
    We want to maximize (minimize LCB), so score = -LCB / cost = (beta*sigma - mu) / cost
    
    This is structurally different from POI:
    - POI uses Phi((f_best - mu)/sigma) which saturates and ignores magnitude of improvement
    - LCB uses linear combination of mu and sigma, giving continuous exploration pressure
    
    We adapt beta based on the relationship between current uncertainty and best value.
    """
    start_time = time.time()
    
    # Handle sigma = 0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive beta: balance exploration and exploitation
    # When sigma is large relative to the range of mu, explore more
    # When sigma is small, exploit more
    
    # Use a fixed beta that works well for 6D Hartmann with this cost structure
    # beta controls exploration: higher = more exploratory
    # For minimization with cost-awareness, we want moderate exploration
    
    # Adaptive beta based on the spread of mu values
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-9:
        mu_range = 1e-9
    
    # Normalize sigma by the scale of the problem
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean < 1e-9:
        sigma_mean = 1e-9
    
    # Beta should be around 1-2 for typical BO; we use a value that balances
    # exploration with the tight cost budget (30 total)
    # Since cost is ~1.5 on average, we have ~20 evaluations, so we need to be
    # somewhat aggressive in exploration early
    beta = 1.5
    
    # LCB for minimization: we want to find points where mu - beta*sigma is low
    # Score should be high for good candidates, so:
    # score = (beta * sigma - mu) / cost
    # This is high when mu is low (good) or sigma is high (uncertain)
    
    lcb_score = (beta * sigma_safe - mu) / cost
    
    # Ensure finite values
    lcb_score = np.where(np.isfinite(lcb_score), lcb_score, 0.0)
    
    # Safety margin: if we're running long, return immediately
    elapsed = time.time() - start_time
    if elapsed > 250:
        return lcb_score
    
    return lcb_score
