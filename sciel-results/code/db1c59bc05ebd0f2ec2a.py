import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling: sample from the posterior and return negative samples as scores
    # This naturally balances exploration and exploitation
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    scores = -samples
    return scores
