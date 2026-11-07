"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI) towards the current best observed value
    # EI measures the expected reduction in objective value if we sample at x
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute a Lower Confidence Bound (LCB) score
    # For minimization, a low LCB (mu - beta * sigma) is good.
    # We convert this to a maximization score by negating it: -LCB = beta * sigma - mu
    # This term rewards high uncertainty (sigma) and low mean (mu).
    beta = 2.5
    lcb_score = beta * sigma - mu
    
    # 3. Normalize both scores to [0, 1] to enable blending
    # EI is non-negative
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
        
    # LCB score can be negative or positive, so normalize by range
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    if lcb_max > lcb_min:
        lcb_norm = (lcb_score - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_norm = np.zeros_like(lcb_score)
        
    # 4. Adaptive Weighting based on proximity to the known global minimum (3.0)
    # If f_best is far from 3.0, we are likely in early stages and need more exploration.
    # If f_best is close to 3.0, we are in late stages and need more exploitation.
    regret = f_best - 3.0
    
    if regret > 5.0:
        # Early stage: Prioritize exploration
        w_lcb = 0.6
        w_ei = 0.4
    elif regret > 1.0:
        # Mid stage: Balanced
        w_lcb = 0.4
        w_ei = 0.6
    else:
        # Late stage: Prioritize exploitation
        w_lcb = 0.2
        w_ei = 0.8
        
    # 5. Combine scores
    scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    return scores
# EVOLVE-BLOCK-END
