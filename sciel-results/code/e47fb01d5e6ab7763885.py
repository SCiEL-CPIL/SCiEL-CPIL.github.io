"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_ACQ_START_TIME = time.time()
_ACQ_DEADLINE = _ACQ_START_TIME + 280


def acquisition(mu, np_sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rosenbrock 2d.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - np_sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: scalar, best observed value so far
    - cost: array of shape (2000,), evaluation costs for each candidate
    
    Returns:
    - array of shape (2000,), acquisition scores (higher = better)
    """
    # Deadline guard
    if time.time() > _ACQ_DEADLINE:
        fallback = -mu / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(fallback), fallback, 0.0)

    # Sanitize
    sigma = np.maximum(np_sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)

    # Standardize f_best relative to current scale for numerical stability
    # Use a robust reference: if f_best is 0, use a small epsilon
    ref = max(abs(f_best), 1e-6)
    z = (f_best - mu) / sigma

    # Compute EI with numerical stability
    ei = np.zeros_like(mu)
    mask = np.abs(z) < 20
    z_clipped = np.clip(z, -20, 20)

    # Standard normal CDF and PDF
    cdf_vals = norm.cdf(z_clipped[mask])
    pdf_vals = norm.pdf(z_clipped[mask])

    ei[mask] = (f_best - mu[mask]) * cdf_vals + sigma[mask] * pdf_vals

    # For very large positive z, EI ≈ f_best - mu
    large_pos = (~mask) & (z > 0)
    ei[large_pos] = f_best - mu[large_pos]

    # For very negative z, EI ≈ 0
    ei[(~mask) & (z <= 0)] = 0.0

    ei = np.maximum(ei, 0.0)

    # Cost-aware score with exploration-exploitation balance
    # Use a blend: primarily EI/cost, but add a small exploratory term
    # that favors regions with high sigma (uncertainty) to avoid premature convergence
    # This is like a soft UCB combined with cost-aware EI
    
    # Normalize sigma for the exploration term
    sigma_norm = sigma / (sigma.max() + 1e-12)
    
    # Cost-aware EI
    ei_cost = ei / cost
    
    # Exploration bonus: scaled by cost efficiency
    # Use a tunable factor; here we use a modest exploration weight
    exploration_weight = 0.15
    exploration = exploration_weight * sigma_norm / cost
    
    # Combined score
    score = ei_cost + exploration
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
