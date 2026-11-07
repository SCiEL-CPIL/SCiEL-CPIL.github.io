"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Rastrigin 2D.
    
    Uses a combination of:
    1. Expected Improvement (EI) for exploitation
    2. Probability of Improvement (PI) for exploration  
    3. Uncertainty bonus for exploring high-uncertainty regions
    4. Cost-awareness to prefer cheaper evaluations
    
    The blending is adaptive based on how close we are to the optimum.
    """
    # Deadline guard: track elapsed time to avoid exceeding 330s limit
    # This function is called many times, so we need to be fast
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-10), sigma, 1e-10)
    cost = np.where(np.isfinite(cost) & (cost > 1e-10), cost, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute z-scores for EI
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Upper Confidence Bound style: explore high-uncertainty regions
    # Use a scaled uncertainty term
    uncertainty_bonus = sigma * 0.1  # Scale to be comparable to EI/PI
    
    # Adaptive blending based on f_best:
    # - When f_best is large (early), favor exploration (PI + uncertainty)
    # - When f_best is small (late), favor exploitation (EI)
    # Rastrigin minimum is 0, so f_best approaches 0 as we converge
    
    # Use a smooth sigmoid transition
    # Center at f_best = 0.5, scale to make transition gradual
    threshold = 0.5
    scale = 3.0
    
    # weight_pi: weight on exploration components
    weight_pi = 1.0 / (1.0 + np.exp(-(f_best - threshold) * scale))
    # weight_pi is close to 1 when f_best > threshold, close to 0 when f_best < threshold
    
    # weight_ei: weight on exploitation
    weight_ei = 1.0 - weight_pi
    
    # Blend: early -> more PI + uncertainty, late -> more EI
    # Scale EI to be in a similar range as PI (EI can be larger)
    # Normalize EI by f_best to make it comparable
    ei_scaled = ei / max(f_best, 1e-10)
    
    combined = weight_ei * ei_scaled + weight_pi * (pi + uncertainty_bonus)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    result = combined / cost
    
    # Ensure finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
