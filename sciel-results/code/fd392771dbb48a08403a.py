"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: normalized blending of Expected Improvement (EI)
    and Lower Confidence Bound (LCB) with adaptive weighting.
    
    The blend adapts based on the relative uncertainty compared to the
    improvement potential, promoting exploration when uncertainty is high
    relative to potential gain, and exploitation when a clear improvement
    is likely.
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Compute Lower Confidence Bound (UCB for minimization: mu - beta * sigma)
    # We want to minimize, so LCB = mu - beta * sigma gives lower bound on f
    # Higher LCB scores mean more likely to find low values
    # Actually for minimization, we want to explore where mu - beta*sigma is low
    # So the "score" for LCB should be -(mu - beta*sigma) = -mu + beta*sigma
    # But we want higher scores to mean "evaluate next"
    # LCB for minimization: pick point with smallest mu - beta*sigma
    # To convert to "higher is better": score = -mu + beta*sigma
    
    # Adaptive beta: larger when uncertainty is high relative to improvement
    # Use a measure of how "promising" the region is
    # When f_best - mu is large relative to sigma, exploitation (EI) dominates
    # When sigma is large relative to f_best - mu, exploration (LCB) helps
    
    # Normalize EI to [0, 1] roughly
    # EI max is when mu << f_best, EI ~ f_best - mu
    # Let's use a sigmoid-based weight
    
    # Compute a measure of exploration need: high sigma relative to potential improvement
    # improvement_potential = max(f_best - mu, 0)
    improvement_potential = np.maximum(f_best - mu, 0)
    
    # When sigma is large compared to improvement potential, explore more
    # ratio = sigma / (improvement_potential + eps)
    # Use this to blend
    
    # Simple adaptive weight:
    # w_lcb = 1 / (1 + improvement_potential / (sigma + eps))
    # When improvement_potential >> sigma: w_lcb ~ sigma/improvement_potential (small) -> exploit
    # When improvement_potential << sigma: w_lcb ~ 1 (large) -> explore
    
    eps = 1e-12
    w_lcb = 1.0 / (1.0 + improvement_potential / (sigma_safe + eps))
    w_ei = 1.0 - w_lcb
    
    # LCB score: higher is better for "pick this point"
    # For minimization, we want low f, so score = -mu + beta * sigma
    # Choose beta adaptively too, or fixed
    beta = 2.0  # exploration-exploitation tradeoff
    lcb_score = -mu + beta * sigma_safe
    
    # Normalize both to comparable scales
    # EI is always >= 0, typically in range [0, f_best - min_mu]
    # LCB score can be negative or positive
    
    # Let's normalize EI by its max possible value
    ei_max = np.max(ei) + eps
    ei_norm = ei / ei_max
    
    # Normalize LCB score
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min + eps
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # Blend
    scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
