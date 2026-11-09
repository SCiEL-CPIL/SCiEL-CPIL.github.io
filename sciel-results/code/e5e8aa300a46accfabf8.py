"""Rewritten acquisition for cost-aware Bayesian optimization on rosenbrock 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of improvement (POI) per unit cost.
    # POI is bounded in [0, 1] and scale-invariant, which provides more stable
    # selection than EI when function magnitudes vary and cost penalties are present.
    # Dividing by cost gives a cost-aware score: higher = more likely improvement
    # per unit cost spent.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    # Ensure finite scores; clip to avoid numerical issues
    poi = np.clip(poi, 1e-12, 1.0 - 1e-12)
    cost_safe = np.maximum(cost, 1e-9)
    return poi / cost_safe
# EVOLVE-BLOCK-END
