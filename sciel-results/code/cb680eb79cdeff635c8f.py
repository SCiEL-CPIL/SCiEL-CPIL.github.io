# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    sigma_safe = np.maximum(sigma, 1e-9)
    diff = f_best - mu
    z = diff / sigma_safe
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    base = diff * cdf_val + sigma_safe * pdf_val
    bonus = sigma_safe / (np.abs(diff) + sigma_safe + 1e-6)
    result = base + 0.05 * bonus
    result[~np.isfinite(result)] = -1e10
    return result
# EVOLVE-BLOCK-END
