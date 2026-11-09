import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # We want to minimize f(x).
    # Posterior predictive: f(x) ~ N(mu(x), sigma(x)^2)
    # Draw a sample from the posterior: sample(x) = mu(x) + sigma(x) * z(x)
    # We want to pick x that minimizes sample(x).
    # The contract requires returning scores where HIGHER is better.
    # So we return -sample(x).
    
    z = np.random.standard_normal(len(mu))
    samples = mu + sigma * z
    # To handle potential numerical issues if sigma is very large or small
    # But standard_normal is fine.
    
    return -samples
