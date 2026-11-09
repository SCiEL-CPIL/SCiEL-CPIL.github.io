"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We aim to minimize the function f(x).
    # The Lower Confidence Bound is L(x) = mu(x) - k * sigma(x).
    # To find the point with the lowest lower bound (most promising for minimization),
    # we maximize the score S(x) = -L(x) = -mu(x) + k * sigma(x).
    
    # A fixed exploration-exploitation trade-off parameter.
    # Values around 1.0 to 2.0 are typical for LCB in Bayesian Optimization.
    k = 1.5
    
    # Compute the score
    score = -mu + k * sigma
    
    # Ensure finite values (though mu and sigma should be finite from GP)
    # If any NaN/Inf appear, replace them with a very low score to avoid selection
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
