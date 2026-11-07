# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI).
    
    Rationale:
    The current strategy (Hybrid EI/LCB) provides smooth gradients but can get
    stuck in local optima in multimodal landscapes like Hartmann 3D. 
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x)).
    
    Unlike EI, which scales the improvement by the magnitude, PI is a probability
    in [0, 1]. It favors points where the posterior mean is well below f_best 
    relative to the uncertainty. 
    
    To prevent pure exploitation (which would just pick the argmin of mu), we 
    add a small exploration term based on sigma, but the core driver is the 
    probability mass of being better than the current best. This creates a 
    "step-like" landscape in the acquisition scores, which can help jump between
    basins if one basin has a slightly higher mean but much lower uncertainty 
    (higher probability of being the best) compared to another basin with a 
    slightly lower mean but high uncertainty.
    
    We use a slightly aggressive prior for the first few steps by boosting the 
    exploration term if the global uncertainty is still high.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: Phi(z)
    pi = norm.cdf(z)
    
    # We want higher scores to be better.
    # PI is already in [0, 1].
    
    # Add an exploration component to avoid getting stuck in a single point
    # if the GP is overconfident.
    # The exploration term is scaled by sigma.
    # We want to explore when sigma is high.
    
    # Normalize sigma to [0, 1] for scaling the exploration bonus
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min
    if sigma_range < 1e-8:
        sigma_norm = np.zeros_like(sigma)
    else:
        sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Adaptive exploration weight:
    # If max_sigma is high (early on), we explore more.
    # If max_sigma is low (late on), we exploit more.
    # max_sigma is in standardized units (roughly [0, 1] initially).
    
    max_sigma = np.max(sigma)
    
    # Weight for exploration term.
    # Early on (max_sigma ~ 1.0), weight is ~ 0.5.
    # Late on (max_sigma ~ 0.1), weight is ~ 0.0.
    weight_explore = np.clip(max_sigma, 0.0, 1.0) * 0.5
    
    # Exploration score: normalized sigma
    explore_score = sigma_norm
    
    # Combined score
    # We scale PI to make it comparable to the exploration term.
    # PI is in [0, 1]. Explore is in [0, 1].
    # If we just sum them, they are on similar scales.
    
    scores = pi + weight_explore * explore_score
    
    # Time check
    if time.time() - start_time > 325:
        # Fallback to a very simple score if we run out of time
        # Just use -mu (exploit best mean)
        return -mu
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
