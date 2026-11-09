import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # Draw samples from the posterior for each candidate
    # y ~ N(mu, sigma^2)
    # We want to minimize y, so we maximize -y
    # Score = -y
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Generate standard normal random variables
    z = np.random.randn(len(mu))
    
    # Sample from the posterior
    y_samples = mu + sigma * z
    
    # Score is the negative of the sampled value (since we minimize)
    scores = -y_samples
    
    # Handle any potential numerical issues
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
