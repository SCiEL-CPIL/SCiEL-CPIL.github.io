# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Lower Confidence Bound (LCB).
    
    For minimization, we want to find points where the lower bound of the 
    posterior distribution is low.
    Lower Bound = mu - beta * sigma.
    We maximize score = -(mu - beta * sigma) = beta * sigma - mu.
    
    The beta parameter is adaptive:
    - When f_best is high (early stages, far from optimum 0), beta is large 
      to encourage exploration of high-uncertainty regions.
    - When f_best is low (late stages, close to optimum 0), beta is smaller 
      to encourage exploitation of low-mean regions.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Adaptive beta:
    # f_best is the lowest observed value. Minimum is 0.
    # Early: f_best ~ 1.0 -> beta ~ 5.0 (explore)
    # Late: f_best ~ 0.0 -> beta ~ 1.0 (exploit)
    # We clamp f_best to [0, 2] to handle potential outliers or weird initializations
    f_best_clamped = np.clip(f_best, 0.0, 2.0)
    beta = 1.0 + 4.0 * f_best_clamped
    
    # Score: Higher is better.
    # LCB for minimization is mu - beta*sigma.
    # We want to minimize LCB, so we maximize -LCB = beta*sigma - mu.
    score = beta * sigma - mu
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
