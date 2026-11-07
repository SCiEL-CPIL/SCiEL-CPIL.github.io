import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB
    # Score = (-mu + beta * sigma) / cost
    
    # Adaptive beta:
    # We want to explore more when sigma is high relative to the improvement potential.
    # A simple robust beta is based on the 90th percentile of sigma.
    # This ensures we don't get overwhelmed by outliers in sigma.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate a dynamic beta
    # If we have very low uncertainty, beta should be small.
    # If we have high uncertainty, beta should be larger.
    # Let's use the ratio of the 95th percentile of sigma to the mean sigma.
    # This normalizes the exploration term.
    if np.std(sigma) > 1e-9:
        beta = np.percentile(sigma, 95) / (np.mean(sigma) + 1e-9)
        # Cap beta to avoid extreme exploration
        beta = np.clip(beta, 1.0, 3.0)
    else:
        beta = 1.0
        
    # The exploitation term is -mu because we want to minimize mu.
    # The exploration term is beta * sigma.
    # We divide by cost to make it cost-aware.
    
    score = (-mu + beta * sigma_safe) / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
