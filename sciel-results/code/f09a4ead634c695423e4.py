import numpy as np

def acquisition(mu, sigma, f_best):
        # Independent Thompson Sampling
        # Sample K values from N(mu, sigma^2) for each candidate
        K = 50
        # Generate random samples: shape (2000, K)
        # Use standard normal and scale
        samples = mu[:, np.newaxis] + sigma[:, np.newaxis] * np.random.normal(size=(len(mu), K))
        
        # For minimization, the "best" sample for a candidate is the minimum
        # We want to maximize the score, so we return -min(samples)
        min_samples = np.min(samples, axis=1)
        
        return -min_samples
