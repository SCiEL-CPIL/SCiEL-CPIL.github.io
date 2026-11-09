import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 2.0  # 2 second safety margin
    
    # Calculate adaptive beta
    mu_spread = np.std(mu)
    if mu_spread < 1e-6:
        mu_spread = 1e-6
        
    sigma_mean = np.mean(sigma)
    if sigma_mean < 1e-6:
        sigma_mean = 1e-6
        
    # Heuristic: beta scales with the ratio of function variation to uncertainty
    # High variation / Low uncertainty -> High beta? 
    # If we are confident (low sigma) but function varies (high mu spread), 
    # we should exploit the best points. So beta should be LOW.
    # So beta should be inversely proportional to mu_spread/sigma_mean?
    # beta = k * (sigma_mean / mu_spread)
    
    # Let's try:
    # If sigma_mean is high (uncertain), beta should be high (explore).
    # If mu_spread is high (rugged), beta should be ?
    
    # Let's stick to the hint: "scales with the ratio of sigma to mu spread"
    # Ratio = sigma / mu_spread.
    # If Ratio is high, beta is high.
    # beta = k * (sigma_mean / mu_spread)
    
    beta = 0.5 * (sigma_mean / mu_spread)
    beta = np.clip(beta, 0.1, 5.0)
    
    # UCB Score: Maximize -mu - beta * sigma
    score = -mu - beta * sigma
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple EI if we are running out of time (unlikely)
        z = (f_best - mu) / (sigma + 1e-9)
        from scipy.stats import norm
        score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
    return score
