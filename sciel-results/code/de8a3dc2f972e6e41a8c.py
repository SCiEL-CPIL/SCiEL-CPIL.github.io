"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values in inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-12), cost, 1.0)
    
    # Clip sigma to avoid division by zero and extreme values
    sigma = np.clip(sigma, 1e-8, 1e6)
    
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-8)
    
    # Deadline guard: track elapsed time
    t_start = time.time()
    
    # Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_val + sigma * pdf_val
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Check deadline before adding exploration bonus
    if time.time() - t_start > 280:
        # Fallback: just return EI/cost
        score = ei / cost
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Bounded exploration bonus: sigma/(sigma + 0.1)
    # This is bounded in [0, 1) and prevents exploration from dominating
    # when sigma is large. The scale 0.1 is chosen to be small relative to
    # the expected objective range of Levy 2d (~[0, 20+]).
    exploration_bonus = sigma / (sigma + 0.1)
    
    # Combine EI and exploration with a small weight
    # alpha is kept small so that EI dominates the decision
    alpha = 0.05
    combined = ei + alpha * exploration_bonus
    
    # Cost-aware acquisition: combined score per unit cost
    score = combined / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final deadline check
    if time.time() - t_start > 290:
        # This should not happen, but just in case
        return score
        
    return score
# EVOLVE-BLOCK-END
