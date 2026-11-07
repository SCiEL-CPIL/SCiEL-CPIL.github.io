import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    sigma = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma
    # Probability of Improvement
    # We want to maximize the probability that f(x) < f_best
    # P(f(x) < f_best) = P((f(x) - mu)/sigma < (f_best - mu)/sigma) = Phi(z)
    scores = norm.cdf(z)
    return scores
