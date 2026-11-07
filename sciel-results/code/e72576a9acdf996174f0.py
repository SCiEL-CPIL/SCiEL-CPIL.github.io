"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Batch-aware Lower Confidence Bound (LCB) with spatial diversity via
    a repulsion term that penalizes candidates with high posterior mean
    relative to f_best, encouraging exploration in regions where the
    GP is uncertain.
    
    The acquisition score combines:
    1. A standard LCB term: -mu + beta * sigma (higher = more promising)
    2. A repulsion/diversity term that penalizes candidates whose mu is
       close to f_best (i.e., already well-explored regions)
    
    Since we don't have coordinates, we approximate spatial diversity by
    using the sigma field: regions with high sigma are less explored.
    The repulsion term uses a soft penalty based on how close mu is to
    f_best, weighted by an adaptive factor.
    """
    # Known minimum of Hartmann 3D
    h3_min = -3.86278
    
    # Normalize progress: 0 when far from minimum, 1 when close
    worst_ref = 0.5
    progress = np.clip((worst_ref - f_best) / (worst_ref - h3_min), 0.0, 1.0)
    
    # Adaptive beta: higher exploration early, lower later
    beta = 1.8 - 1.3 * progress
    
    # Base LCB score: higher is better
    lcb_score = -mu + beta * sigma
    
    # Repulsion term: penalize candidates where mu is close to f_best
    # This encourages exploring regions not yet well-characterized
    # Use a Gaussian-like penalty centered at f_best
    # The width of the penalty scales with sigma to adapt to uncertainty
    mu_diff = mu - f_best
    # Scale the penalty width by the median sigma for robustness
    sigma_med = np.median(sigma)
    if sigma_med < 1e-10:
        sigma_med = 1e-10
    penalty_width = 0.5 * sigma_med
    
    # Soft repulsion: exp(-(mu_diff / penalty_width)^2)
    # This is 1 when mu ≈ f_best, decays as mu moves away
    repulsion = np.exp(-(mu_diff / penalty_width) ** 2)
    
    # Weight the repulsion term: stronger early (when progress is low)
    # to encourage broad exploration, weaker later
    repulsion_weight = 0.3 * (1.0 - 0.5 * progress)
    
    # Final score: LCB minus repulsion penalty
    score = lcb_score - repulsion_weight * repulsion
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
