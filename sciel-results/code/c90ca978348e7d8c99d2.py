"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math

# Global counter for dynamic beta
_t = 0

def acquisition(mu, sigma, f_best):
    global _t
    _t += 1
    
    # Dynamic UCB beta
    # Theoretical bound for GP-UCB
    # beta_t = sqrt(2 * log(1 + t * pi^2 / (6 * log(1 + t))))
    # For simplicity and robustness, use a simpler dynamic form
    # beta_t = sqrt(2 * log(1 + t))
    
    # To avoid zero sigma issues, ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Calculate beta
    beta = math.sqrt(2.0 * math.log(1.0 + _t))
    
    # For minimization, we use Lower Confidence Bound (LCB)
    # LCB = mu - beta * sigma
    # We want to minimize LCB.
    # The task requires returning scores where HIGHER means evaluate next.
    # So we return -LCB = -mu + beta * sigma
    
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
