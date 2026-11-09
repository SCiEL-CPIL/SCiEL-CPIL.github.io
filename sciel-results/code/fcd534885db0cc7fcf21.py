"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound with a tuned exploration coefficient
    # For Goldstein-Price 2D, a higher exploration coefficient helps escape local optima
    # in the multimodal landscape. We use a slightly larger coefficient than typical.
    beta = 2.0
    scores = mu + beta * sigma
    return scores
# EVOLVE-BLOCK-END
