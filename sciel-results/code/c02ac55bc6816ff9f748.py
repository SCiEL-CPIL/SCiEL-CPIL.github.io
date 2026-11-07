# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Regret: P(f < f_best) = Phi((f_best - mu) / sigma).
    # A structurally different objective from EI: it rewards the *probability*
    # of finding a new best rather than the expected *magnitude* of improvement.
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against degenerate / negative std (sigma is a standard deviation).
    s = np.clip(sigma, 1e-9, None)

    z = (f_best - mu) / s
    z = np.clip(z, -38.0, 38.0)  # keep CDF finite and well-behaved

    # Higher score == more likely to be a new best == evaluate next.
    scores = norm.cdf(z)
    
    # Add a small exploration bonus to encourage visiting uncertain regions.
    # This helps escape local minima by slightly favoring high-uncertainty points
    # while keeping the probability of improvement as the primary signal.
    # The bonus is scaled to be small relative to the CDF range [0, 1].
    scores += 0.05 * s / (np.max(s) + 1e-12)
    
    return scores
# EVOLVE-BLOCK-END
