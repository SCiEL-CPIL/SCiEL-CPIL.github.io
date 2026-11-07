import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Use a small negative xi to encourage exploration in early stages
    # This is a refinement of the Expected Improvement family.
    xi = -0.1
    target = f_best - xi
    z = (target - mu) / (sigma + 1e-9)
    # Calculate Expected Improvement
    scores = (target - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    return scores
