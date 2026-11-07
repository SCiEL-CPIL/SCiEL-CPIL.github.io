"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize f(x).
    # The lower confidence bound is L(x) = mu(x) - beta * sigma(x)
    # We select the point with the lowest L(x), which is equivalent to
    # maximizing the score S(x) = -L(x) = -mu(x) + beta * sigma(x)
    
    # Choose beta to balance exploration and exploitation.
    # For a small budget (30 evals) in a multimodal 2D landscape,
    # a higher beta encourages exploration of high-variance regions.
    beta = 1.5
    
    # Calculate score
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    scores = -mu + beta * sigma_safe
    
    return scores
# EVOLVE-BLOCK-END
