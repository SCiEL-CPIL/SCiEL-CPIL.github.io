"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with strict penalty on uncertainty.
    
    Uses a modified Expected Improvement where the uncertainty term is
    heavily penalized to prevent domination by high-noise regions in
    early sparse data regimes.
    """
    # Small epsilon to avoid division by zero
    eps = 1e-9
    
    # Standardize sigma for numerical stability
    sigma_safe = np.maximum(sigma, eps)
    
    # z-score for standard normal CDF/PDF
    z = (f_best - mu) / sigma_safe
    
    # Standard EI components
    ei_mean = (f_best - mu) * norm.cdf(z)
    ei_uncertainty = sigma_safe * norm.pdf(z)
    
    # Strict penalty on uncertainty: scale uncertainty by a factor that
    # decreases rapidly with sigma magnitude. This forces the acquisition
    # function to prioritize regions where the improvement is highly certain
    # (low sigma) over regions with high potential but high risk.
    # We use a stronger decay rate (alpha=5.0) compared to the previous 1.5
    # to more aggressively suppress exploration in uncertain areas.
    alpha = 5.0
    uncertainty_penalty = np.exp(-alpha * sigma_safe)
    
    # Modified EI: mean improvement + penalized uncertainty
    modified_ei = ei_mean + uncertainty_penalty * ei_uncertainty
    
    # Ensure non-negative scores
    modified_ei = np.maximum(modified_ei, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    cost_safe = np.maximum(cost, eps)
    scores = modified_ei / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
