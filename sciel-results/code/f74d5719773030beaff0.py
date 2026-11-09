"""Cost-aware acquisition for three-hump-camel 2D BO using POI with cost penalty."""
import time
import numpy as np
from scipy.stats import norm


def _norm_cdf(x):
    """Vectorized normal CDF."""
    return norm.cdf(x)


def _norm_pdf(x):
    """Vectorized normal PDF."""
    return norm.pdf(x)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Probability of Improvement with adaptive
    cost weighting.
    
    Parameters
    ----------
    mu : array_like, shape (2000,)
        GP mean predictions for 2000 candidates, in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions for 2000 candidates, in objective units.
    f_best : float
        Best observed function value (lower is better).
    cost : array_like, shape (2000,)
        Evaluation cost for each candidate point.
    
    Returns
    -------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    start_time = time.time()
    deadline = start_time + 280.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)

    # Compute z-score for POI
    z = (f_best - mu) / sigma
    z = np.clip(z, -15, 15)

    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    poi = _norm_cdf(z)

    # Also compute EI for a blended approach
    cdf_z = poi
    pdf_z = _norm_pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)

    # Normalize costs to [0, 1]
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    if cost_max > cost_min:
        cost_norm = (cost_safe - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost_safe)

    # Adaptive convergence weight: as we get closer to the optimum,
    # we should be more cost-sensitive
    # Use the best mu as a proxy for how close we are
    mu_min = np.min(mu)
    # If f_best is close to the theoretical min (0), increase cost sensitivity
    convergence_factor = 1.0 - np.exp(-f_best / 1.0) if f_best > 0 else 1.0
    convergence_factor = np.clip(convergence_factor, 0.3, 1.0)

    # Cost penalty: multiplicative, stronger when cost is high
    # alpha adapts based on convergence
    alpha = 0.5 * convergence_factor
    cost_penalty = 1.0 - alpha * cost_norm

    # Blend POI and EI for robustness
    # POI is smoother in low-uncertainty regions, EI captures magnitude
    # Weight based on relative sigma
    sigma_max = np.max(sigma) + 1e-10
    sigma_rel = sigma / sigma_max
    
    # When sigma is high, trust EI more (it captures expected improvement magnitude)
    # When sigma is low, trust POI more (it's smoother)
    blend_weight = 0.5 + 0.5 * sigma_rel  # range [0.5, 1.0]
    
    # Normalize EI to be comparable to POI (which is in [0, 1])
    ei_max = np.max(ei) + 1e-10
    ei_norm = ei / ei_max

    blended = blend_weight * ei_norm + (1.0 - blend_weight) * poi

    # Apply cost penalty
    score = blended * cost_penalty

    # Add a small exploration bonus for high uncertainty in unexplored regions
    # This helps break ties and encourages exploration when EI/POI are similar
    sigma_max_val = np.max(sigma) + 1e-10
    exploration_bonus = 0.01 * (sigma / sigma_max_val) * (1.0 - cost_norm)
    score += exploration_bonus

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    # Time guard
    if time.time() > deadline:
        pass

    return score
