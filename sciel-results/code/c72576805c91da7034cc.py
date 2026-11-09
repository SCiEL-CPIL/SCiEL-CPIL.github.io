import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    z = (f_best - mu) / (sigma + 1e-9)
    # Use the LBEI formula derived for minimization
    term1 = (f_best - mu) * (1 - norm.cdf(z))
    term2 = sigma * norm.pdf(z)
    term3 = 0.5 * sigma**2 * (z / (1 + z**2))
    return term1 + term2 + term3
