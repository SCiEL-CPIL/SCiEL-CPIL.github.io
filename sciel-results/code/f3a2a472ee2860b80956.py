import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Strategy: Cost-Weighted Probability of Improvement (CW-PI).
    
    Probability of Improvement (PI) is an acquisition function that focuses on the 
    probability that a candidate point will improve upon the current best observation.
    Unlike Expected Improvement (EI), which considers the magnitude of improvement, 
    PI is simpler and can be more robust in scenarios where the cost of evaluation 
    is a significant factor.
    
    The cost-aware aspect is introduced by dividing the PI by the evaluation cost. 
    This transforms the objective to maximize "Probability of Improvement per Unit Cost".
    Given that the cost is correlated with the distance from the origin (where the 
    minimum is located), this naturally biases the search towards regions that are 
    likely to be good (high probability of improvement) and cheap (close to the minimum).
    
    Formula:
    PI(x) = Phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    Score(x) = PI(x) / cost(x)
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety checks
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e6, neginf=1e-8)
    cost = np.nan_to_num(cost, nan=1.5, posinf=2.0, neginf=1.0)
    
    # Ensure sigma and cost are strictly positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1.0) # Min cost is 1.0
    
    # Calculate Probability of Improvement
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in Phi for extreme values
    z_clipped = np.clip(z, -10, 10)
    
    pi = norm.cdf(z_clipped)
    
    # Cost-weighted score
    score = pi / cost
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Fallback: simple exploitation based on cost-adjusted mean
        score = -mu / cost
        
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
