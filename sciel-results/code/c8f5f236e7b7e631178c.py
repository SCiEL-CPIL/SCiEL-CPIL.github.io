"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a Knowledge Gradient-inspired approach with exploitation bias.
    """
    # Deadline guard: ensure we return within time limits
    t_start = time.time()
    DEADLINE = 290.0  # Safety margin from 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard normal CDF and PDF
    z = (f_best - mu) / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Knowledge Gradient approximation:
    # The probability that this point improves the best is P(mu < f_best)
    # weighted by the expected reduction if it does improve.
    # We combine EI with a discovery probability term.
    discovery_prob = cdf  # P(mu < f_best)
    
    # Exploitation bias: favor lower mu values
    # Scale mu to [0,1] for consistent bias
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min + 1e-12
    mu_scaled = (mu - mu_min) / mu_range
    
    # Strong exploitation bias toward lower values
    exploitation = -0.15 * mu_scaled
    
    # Uncertainty bonus: moderate sigma is best for exploration
    # Use a Gaussian-like bump centered at moderate uncertainty
    sigma_median = np.median(sigma_safe)
    sigma_norm = sigma_safe / (sigma_median + 1e-12)
    uncertainty_bonus = 0.1 * np.exp(-0.5 * (np.log1p(sigma_norm) ** 2))
    
    # Cost adjustment: divide by cost to get per-unit-cost value
    cost_safe = np.maximum(cost, 1e-8)
    
    # Combine terms
    # Base: EI per unit cost
    base = ei / cost_safe
    
    # Add exploitation and uncertainty terms (also cost-adjusted)
    exploitation_cost = exploitation / cost_safe
    uncertainty_cost = uncertainty_bonus / cost_safe
    
    # Weighted combination
    score = base + 0.3 * exploitation_cost + 0.2 * uncertainty_cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - t_start > DEADLINE:
        return score
    
    return score
# EVOLVE-BLOCK-END
