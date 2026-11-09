"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Hybrid acquisition function: Expected Improvement + Exploration Bonus.
    # EI captures exploitation potential, while sigma adds exploration pressure.
    # This aims to balance the strengths of both strategies for the multimodal
    # Weierstrass function.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration bonus weight
    lambda_explore = 1.0
    
    scores = ei + lambda_explore * sigma
    return scores
# EVOLVE-BLOCK-END
