"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition function for minimization.
    
    Uses a sigma-normalized exploration coefficient that adapts to the current
    posterior uncertainty distribution, combined with a mild exploitation bias
    toward regions near f_best.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = more promising)
    """
    # Handle potential zero or near-zero sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute the median sigma as a reference scale for normalization
    sigma_median = np.median(sigma_safe)
    if sigma_median < 1e-12:
        sigma_median = 1e-9
    
    # Adaptive exploration coefficient:
    # - Base kappa of 3.5 for strong exploration
    # - Modulated by the fraction of candidates with high uncertainty
    # This creates a dynamic balance: when most candidates have low sigma
    # (late in optimization), we exploit more; when many have high sigma
    # (early in optimization), we explore more.
    
    # Fraction of candidates with sigma above the median
    high_unc_frac = np.mean(sigma_safe > sigma_median)
    
    # Base exploration coefficient
    kappa_base = 3.5
    
    # Adaptive modulation: scale exploration by how much uncertainty remains
    # When high_unc_frac is ~0.5 (balanced), kappa = kappa_base
    # When high_unc_frac is low (little uncertainty), kappa decreases slightly
    # When high_unc_frac is high (lots of uncertainty), kappa increases slightly
    kappa = kappa_base * (0.8 + 0.4 * high_unc_frac)
    
    # LCB for minimization: mu - kappa * sigma
    lcb = mu - kappa * sigma_safe
    
    # Add a mild exploitation term: prefer points where mu is close to f_best
    # This helps fine-tune near known good regions without sacrificing exploration
    # The term is scaled by sigma so it only matters when uncertainty is low
    exploitation_weight = 0.3
    exploitation_term = -exploitation_weight * np.exp(-0.5 * ((mu - f_best) / (sigma_safe + 1e-12)) ** 2)
    
    # Combine: the exploitation term adds a small bonus when mu is near f_best
    # and sigma is small, but doesn't override the LCB when sigma is large
    scores = -lcb + exploitation_term * (sigma_safe / sigma_median)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
