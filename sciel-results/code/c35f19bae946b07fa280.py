# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Expected Improvement per unit cost,
    with a small exploration bonus to avoid premature convergence.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Standard normal CDF and PDF
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)

    # Expected Improvement (EI)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z

    # Cost-aware EI: EI divided by cost
    cost_aware_ei = ei / cost_safe

    # Exploration bonus: favor points with higher uncertainty (sigma)
    # This helps escape local optima in rastrigin
    # Scale the bonus to be comparable to the EI term
    # The typical EI value is on the order of sigma, so we scale by sigma
    exploration_bonus = 0.05 * sigma_safe / cost_safe

    # Combine: cost-aware EI + exploration bonus
    scores = cost_aware_ei + exploration_bonus

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
