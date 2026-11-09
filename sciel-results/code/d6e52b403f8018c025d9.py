"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing Hartmann 3D.
    Uses a hybrid UCB + EI approach with adaptive exploration.
    
    The UCB term mu - beta * sigma encourages exploration in uncertain regions.
    The EI term provides exploitation pressure toward known good values.
    We blend them with a schedule that shifts from exploration to exploitation.
    """
    # Deadline guard: if we've been running too long, just return EI
    # (though this function should be fast, this is a safety net)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute EI (Expected Improvement) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Compute UCB for minimization: lower is better
    # UCB = mu - beta * sigma
    # We need to estimate the number of evaluations to set beta.
    # Since we don't have direct access to n_evals, we can infer it from
    # the scale of sigma or use a fixed schedule based on the function's behavior.
    # 
    # A common heuristic: beta increases with sqrt(n) but we don't have n.
    # Instead, we can use the ratio of sigma to the range of mu to adapt.
    # Or simply use a fixed beta that works well for Hartmann 3D.
    #
    # From the inspiration, UCB with adaptive beta works well.
    # Let's use a beta that starts high and decays.
    # Since we don't know n, we can use the fraction of points where sigma is high
    # as a proxy for "early in the search" when uncertainty is high everywhere.
    
    # Normalize EI and UCB to comparable scales
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    # For UCB, we want to balance exploration.
    # A simple approach: use beta = 1.0 * sqrt(log(n)/n) but we don't have n.
    # Let's use a fixed beta that has been shown to work well.
    # For Hartmann 3D with 30 evaluations, a beta around 0.5-1.0 works.
    
    # Use the standard deviation of sigma as a proxy for exploration need
    # When sigma is high overall, we're early in the search -> explore more
    sigma_mean = np.mean(sigma_safe)
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-12:
        mu_range = 1e-12
    
    # Relative uncertainty: if sigma is large compared to mu range, explore more
    rel_uncertainty = sigma_mean / mu_range
    
    # Beta scales with relative uncertainty, capped to avoid excessive exploration
    beta = np.clip(0.5 * rel_uncertainty + 0.1, 0.1, 1.5)
    
    ucb = mu - beta * sigma_safe
    
    # Normalize UCB
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-12:
        ucb_norm = np.zeros_like(ucb)
    else:
        ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Blend EI and UCB
    # Early on (high uncertainty), weight UCB more
    # Later (low uncertainty), weight EI more
    # Use rel_uncertainty as a proxy for the schedule
    ucb_weight = np.clip(0.3 * rel_uncertainty, 0.0, 0.5)
    ei_weight = 1.0 - ucb_weight
    
    scores = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
