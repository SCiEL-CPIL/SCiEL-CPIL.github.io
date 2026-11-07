"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware expected improvement with a mild exploration term.
    # cost in [1, 2]; lower cost => higher score (cheaper to evaluate).
    eps = 1e-9
    s = np.maximum(sigma, eps)
    z = (f_best - mu) / s
    # Clip z to avoid overflow in cdf/pdf.
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    # Cost normalization: divide by cost so cheap points are favored.
    c = np.maximum(cost, eps)
    score = ei / c
    # Small exploration term: reward uncertainty in cheap regions to avoid
    # premature convergence. Scaled down so EI dominates when confident.
    score = score + 0.05 * s / c
    return score
# EVOLVE-BLOCK-END
