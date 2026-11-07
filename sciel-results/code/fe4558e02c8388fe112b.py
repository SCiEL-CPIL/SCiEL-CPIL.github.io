"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Check deadline: 330s limit, leave 10s margin
    if time.time() - _START_TIME > 320:
        # Return a neutral score if we are running out of time
        # Ideally this shouldn't happen, but it's a safety net
        return np.zeros(len(mu))

    # Ensure inputs are valid and 1D
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero in Z calculation
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # f_best is the lowest observed value
    # For minimization, Expected Improvement (EI) is:
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    # where Phi is CDF and phi is PDF of standard normal distribution.
    
    # Calculate Z-score
    z = (f_best - mu) / sigma
    
    # Calculate EI
    # Use scipy.stats.norm for Phi and phi
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors might cause slight negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Awareness: 
    # We want to maximize EI per unit cost.
    # Simple ratio: EI / cost
    # This encourages finding improvements in cheaper regions if the EI is similar.
    scores = ei / cost
    
    # Replace any non-finite values with 0
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
