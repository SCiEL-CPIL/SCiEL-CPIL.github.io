"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Explicit wall-clock deadline guard (evaluator kills at 330 s; keep margin).
_DEADLINE = time.time() + 325.0

# Pre-compute the z-score for the 95th percentile (1.64485)
# This corresponds to the standard normal quantile for 0.95
_Z_95 = norm.ppf(0.95)

def acquisition(mu, sigma, f_best):
    # Time guard: if we are too close to the deadline, fall back to pure exploitation
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=float)
        return -mu
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Robustify sigma: clip tiny values and drop non-finite entries
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.clip(sigma, 1e-9, None)
    
    # Quantile-based Expected Improvement (qEI)
    # We want to maximize the expected improvement of the quantile prediction over f_best.
    # Let Y ~ N(mu, sigma^2).
    # We define the improvement I = max(0, f_best - Y_q) where Y_q = mu + z * sigma.
    # However, a more direct acquisition function is the "Expected Quantile Improvement".
    # Alternatively, we can use a standard EI but on the quantile point.
    
    # A simpler and very effective approach for Ackley (multimodal) is to use
    # a "Quantile EI" which encourages points where the high quantile of the distribution
    # is significantly lower than f_best.
    
    # Score = E[max(0, f_best - (mu + z*sigma))]
    # Let d = (f_best - mu - z*sigma) / sigma = (f_best - mu)/sigma - z
    # E[max(0, f_best - Y_q)] = sigma * (d * Phi(d) + phi(d))
    # Wait, the standard EI formula is:
    # E[max(0, f_best - Y)] where Y ~ N(mu, sigma^2)
    # = (f_best - mu) * Phi(d) + sigma * phi(d)
    # where d = (f_best - mu) / sigma
    
    # If we use the quantile Y_q = mu + z*sigma as the "prediction",
    # and assume the uncertainty around this quantile is still sigma (approximation),
    # we can compute EI on this shifted mean.
    # New mean m_q = mu + z*sigma
    # Improvement I = max(0, f_best - Y) where Y ~ N(m_q, sigma^2)
    # d_q = (f_best - m_q) / sigma = (f_best - mu - z*sigma) / sigma
    
    d_q = (f_best - mu - _Z_95 * sigma) / sigma
    d_q = np.where(np.isfinite(d_q), d_q, 0.0)
    
    # Standard EI formula with d_q
    # EI = sigma * (d_q * Phi(d_q) + phi(d_q))
    # We can compute this efficiently using scipy.stats.norm
    # Phi(d_q) = norm.cdf(d_q)
    # phi(d_q) = norm.pdf(d_q)
    
    cdf = norm.cdf(d_q)
    pdf = norm.pdf(d_q)
    
    score = sigma * (d_q * cdf + pdf)
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
