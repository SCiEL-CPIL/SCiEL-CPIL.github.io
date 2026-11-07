import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling with cost normalization
    # Sample from the GP posterior
    samples = mu + sigma * np.random.randn(len(mu))
    # For minimization, lower samples are better
    # Divide by cost to penalize expensive evaluations
    scores = -samples / cost
    return scores
