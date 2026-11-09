import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # This approach samples from the posterior distribution and selects
    # the candidate with the best (lowest) sample.
    # We use a local random number generator to ensure determinism
    # and avoid side effects on the global random state.
    
    # Initialize a local RNG with a fixed seed
    rng = np.random.default_rng(42)
    
    # Generate standard normal random numbers for each candidate
    # Shape matches mu and sigma (2000,)
    z = rng.standard_normal(mu.shape)
    
    # Draw samples from the Gaussian posterior N(mu, sigma^2)
    # sigma is the standard deviation
    samples = mu + sigma * z
    
    # The goal is minimization.
    # Lower sample values indicate better candidates.
    # The acquisition score must be higher for better candidates.
    # Therefore, we return the negative of the samples.
    scores = -samples
    
    return scores
