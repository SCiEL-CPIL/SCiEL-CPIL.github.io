import numpy as np
from scipy.stats import norm
import time

# Global state to track time? 
# The function is stateless in the signature. 
# We can't easily track elapsed time across calls without a global variable.
# However, the task says "Each evaluation must finish within 300s".
# The whole loop is 30 steps. 
# If each step is < 1s, we are safe.

def acquisition(mu, sigma, f_best):
    """
    Adaptive LCB acquisition function for Bayesian optimization.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive exploration parameter beta
    # High beta for exploration when f_best is large (poor solution)
    # Low beta for exploitation when f_best is small (good solution)
    
    # f_best is the current minimum objective value.
    # For sphere 1d on [-5, 5], f_best is in [0, 25].
    
    # Use a smooth step function to transition beta.
    # Center the transition around f_best = 5.0
    # beta ranges from ~1.2 to ~3.0
    
    x_shift = f_best - 5.0
    # Clip to avoid overflow in exp
    x_clip = np.clip(x_shift, -50, 50)
    sigmoid_val = 1.0 / (1.0 + np.exp(-x_clip))
    
    beta = 1.2 + 1.8 * sigmoid_val
    
    # Lower Confidence Bound for minimization:
    # We want to minimize f(x).
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to select the point with the lowest LCB.
    # The evaluator selects the point with the HIGHEST acquisition score.
    # So we return -LCB(x) = -mu(x) + beta * sigma(x)
    
    score = -mu + beta * sigma
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
