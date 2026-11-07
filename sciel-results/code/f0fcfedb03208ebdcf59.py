"""Rewritten program with a hybrid EI/UCB cost-aware acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a neutral score (e.g., based on cost only) to finish gracefully
        return 1.0 / cost

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi = norm.pdf(z)
    cdf = norm.cdf(z)
    ei = (f_best - mu) * cdf + sigma_safe * phi
    
    # Calculate Lower Confidence Bound (LCB) component
    # We want to minimize mu - beta * sigma
    # Score should maximize this, so we use -mu + beta * sigma
    
    # Adaptive Beta:
    # If f_best is close to 0 (optimal), reduce exploration (lower beta).
    # If f_best is high, increase exploration (higher beta).
    # Griewank min is 0. Typical good values < 0.01.
    # Map f_best in [0, 0.1] to beta in [0.5, 2.0]
    # If f_best > 0.1, beta = 2.0
    # If f_best < 0, beta = 0.5
    beta = 0.5 + 1.5 * np.clip(f_best / 0.1, 0.0, 1.0)
    
    lcb_term = -mu + beta * sigma_safe
    
    # Hybrid Score:
    # Combine EI and LCB.
    # EI balances exploration/exploitation via the normal distribution.
    # LCB explicitly adds an exploration bonus.
    # Weights: 0.5 for EI, 0.5 for LCB.
    # This structure was inspired by the successful hybrid in the knowledge base.
    
    score = (0.5 * ei + 0.5 * lcb_term) / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
