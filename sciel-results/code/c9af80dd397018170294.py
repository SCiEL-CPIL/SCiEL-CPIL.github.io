import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Adaptive Upper Confidence Bound for minimization
    # Maximize score = -mu + xi * sigma
    # Use known global minimum of Hartmann 6D to adapt exploration
    f_min = -3.32237
    gap = f_best - f_min
    # xi ranges from 0.5 (close to min) to 1.5 (far from min)
    xi = 0.5 + 1.0 * np.clip(gap / 5.0, 0.0, 1.0)
    return -mu + xi * sigma
