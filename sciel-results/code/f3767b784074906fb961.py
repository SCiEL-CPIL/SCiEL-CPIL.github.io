import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Weighted Thompson Sampling Acquisition Function.
    
    Draws a sample from the GP posterior for each candidate and selects
    the candidate that minimizes the sampled value per unit cost.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety checks
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e6, neginf=1e-8)
    cost = np.nan_to_num(cost, nan=1.5, posinf=2.0, neginf=1.0)
    
    # Ensure sigma and cost are strictly positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1.0)
    
    # Generate random normal noise for Thompson Sampling
    # Using a fixed seed is bad for BO as it makes the process deterministic and potentially stuck.
    # We should use random noise.
    epsilon = np.random.normal(0, 1, size=mu.shape)
    
    # Sample from the posterior
    # f_sample ~ N(mu, sigma^2)
    f_sample = mu + sigma * epsilon
    
    # We want to minimize f_sample.
    # The acquisition score should be higher for better candidates.
    # So we return negative of the sampled value, adjusted for cost.
    # Score = -f_sample / cost
    
    # To avoid division by zero or near-zero cost issues (though cost >= 1.0),
    # we just divide.
    
    score = -f_sample / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
