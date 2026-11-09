import numpy as np

def acquisition(mu, sigma, f_best):
    # Expected Hypar (EHypR) approximation for minimization.
    # Maximizes E[f_best / f(x)] assuming f(x) > 0.
    # Approximation: E[1/Y] ~ (1/mu) * exp(sigma^2 / (2 * mu^2)) for mu > 0.
    
    epsilon = 1e-8
    mu_safe = np.maximum(mu, epsilon)
    
    # Calculate the exponent term: sigma^2 / (2 * mu^2)
    # Clip to prevent numerical overflow in exp
    ratio = (sigma ** 2) / (2.0 * mu_safe ** 2)
    ratio = np.clip(ratio, 0.0, 50.0)
    
    # Score = f_best * E[1/f(x)]
    scores = (f_best / mu_safe) * np.exp(ratio)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
