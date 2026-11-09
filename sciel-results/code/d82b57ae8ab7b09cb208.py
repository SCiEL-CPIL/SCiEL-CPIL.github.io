"""Cost-aware acquisition (POI-based) for BO on the 2D three-hump camel."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability-of-improvement per unit cost, with a mild exploration
    # bonus and a floored uncertainty to keep scores smooth and finite.
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Floor the cost so the very first (zero-cost) phase stays well defined.
    c = np.maximum(cost, 1e-3)

    # Floor the uncertainty: avoids division blowups and the cost=0 singularity,
    # and keeps the POI gradient smooth where the GP is (near-)certain.
    s = np.maximum(sigma, 1e-3)

    # Probability that the next point improves on the best seen so far.
    z = (f_best - mu) / s
    poi = norm.cdf(z)

    # Mild exploration term: reward points with genuine (bounded) uncertainty.
    explore = 0.1 * s

    score = (poi + explore) / c

    # Guarantee a finite, real-valued output of the right shape.
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
