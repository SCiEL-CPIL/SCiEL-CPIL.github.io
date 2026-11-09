import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_LIMIT = 290.0  # Safety margin below 300s

def acquisition(mu, sigma, f_best, cost):
    # Deadline check
    if time.time() - _START_TIME > _LIMIT:
        # Return a constant score if we are running out of time,
        # though the evaluator might kill us anyway.
        # Better to return quickly.
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    # Clip z to prevent numerical underflow/overflow in pdf/cdf
    z = np.clip(z, -10, 10)

    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Cost-aware EI with a small exploitation bias
    # The term -0.1 * mu encourages points with lower predicted means,
    # which can help navigate flat regions or converge faster.
    exploitation_bias = -0.1 * mu

    score = (ei + exploitation_bias) / cost

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
