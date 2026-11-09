import numpy as np

def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle potential numerical issues
    # sigma should be non-negative, but just in case
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive beta based on f_best
    # As f_best decreases, we reduce exploration
    # Scale factor: Beale function values are roughly in [0, 50]
    # We want beta to be ~5-10 early, ~0.5-1 late
    # beta = 5 * sqrt(f_best + 1) ?
    # If f_best = 20, beta = 5 * sqrt(21) ~ 23 (Too high?)
    # If f_best = 0.1, beta = 5 * sqrt(1.1) ~ 5.2 (Too high?)
    
    # Let's try a softer decay.
    # beta = 2.0 / (1.0 + np.exp(-0.5 * f_best)) ? No, that's sigmoid.
    
    # Simple linear-ish decay?
    # beta = 1.0 + 2.0 * (f_best / (f_best + 1.0))
    # If f_best = 100, beta ~ 3.0
    # If f_best = 0.1, beta ~ 1.2
    # This might be too conservative.
    
    # Let's stick to the sqrt scaling but with a smaller base.
    # beta = 1.5 * np.sqrt(f_best + 0.1)
    # f_best=20 -> beta=1.5*4.5=6.75
    # f_best=0.1 -> beta=1.5*0.45=0.67
    
    beta = 1.5 * np.sqrt(f_best + 0.1)
    
    # LCB for minimization: minimize (mu - beta * sigma)
    # Score for ranking (higher is better): - (mu - beta * sigma)
    score = -mu + beta * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
