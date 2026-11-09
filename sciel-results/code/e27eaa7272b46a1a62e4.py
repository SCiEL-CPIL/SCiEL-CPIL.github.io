"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # UCB acquisition: Exploits high posterior mean (low function value) 
    # and explores high uncertainty (high sigma).
    # Since we are minimizing, we want low mu. 
    # Standard UCB for minimization: score = -mu + beta * sigma
    # We use a fixed beta that balances exploration and exploitation.
    # Given the small number of evaluations (30), a moderate beta is often 
    # superior to EI which can be too conservative or too exploratory.
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Beta parameter: 
    # Too low: pure greedy (exploits current best, may get stuck in local minima)
    # Too high: pure exploration (random walk)
    # For 1D Weierstrass, which is multi-modal but smooth, UCB with beta ~ 1-3 
    # often works well to jump between modes.
    beta = 2.0
    
    # Calculate score: higher is better
    # We want to minimize f, so -mu is the "improvement" direction.
    # Adding beta * sigma encourages sampling where we are uncertain.
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
