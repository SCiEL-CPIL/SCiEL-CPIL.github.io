"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to minimize f(x). The LCB is mu - beta * sigma.
    # The evaluator selects the candidate with the HIGHEST score.
    # Therefore, we return the negative of the LCB to ensure that
    # candidates with low LCB (likely low f) get high scores.
    # Score = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Beta controls exploration. 
    # A value of 2.0 is a standard choice for balancing exploitation and exploration
    # in deterministic Bayesian optimization tasks.
    beta = 2.0
    
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
