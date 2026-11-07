"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Calculates cost-aware Expected Improvement (EI) for each candidate.
    The cost parameter is used to penalize expensive evaluations,
    balancing the trade-off between expected improvement and evaluation cost.
    
    We use a standard EI formulation with numerical stability safeguards,
    adjusted by a cost-awareness factor that reduces the effective EI
    for candidates with higher expected costs.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # Since we are minimizing, improvement is f_best - mu (positive if mu < f_best)
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We compute this in a numerically stable way.
    
    # For large positive z, Phi(z) -> 1, phi(z) -> 0, EI -> f_best - mu
    # For large negative z, Phi(z) -> 0, phi(z) -> 0, EI -> 0
    
    # Use scipy's norm for CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = diff * cdf_z + sigma * pdf_z
    
    # Ensure finiteness and non-negativity (EI should be >= 0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness adjustment:
    # The cost of evaluating a point is 1 + mean(x scaled to [0,1])
    # Since we don't have x coordinates, we use the cost array provided
    # to adjust the EI score. Points with higher cost get a reduced EI.
    # We use a multiplicative factor that reduces EI for expensive points.
    # The factor is 1 / (1 + alpha * (cost - min_cost)) where alpha controls
    # the strength of cost-awareness.
    
    # Normalize cost to [0, 1] range for consistent scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-8:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Apply cost-awareness: reduce EI for expensive candidates
    # alpha = 0.5 means a point with maximum cost gets EI reduced by ~33%
    alpha = 0.5
    cost_factor = 1.0 / (1.0 + alpha * cost_norm)
    
    ei = ei * cost_factor
    
    # Final safety check
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
