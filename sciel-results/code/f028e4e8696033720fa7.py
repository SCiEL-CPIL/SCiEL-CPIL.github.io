import numpy as np

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Adaptive beta: Higher when f_best is poor (early), lower when f_best is good (late)
    # Scale factor 0.1 seems reasonable for Powell function values
    beta = 2.0 * np.tanh(0.1 * f_best)
    
    # Ensure beta doesn't get too small to completely stop exploration
    beta = np.maximum(beta, 0.1)
    
    ucb = -mu + beta * sigma_safe
    score = ucb / cost_safe
    
    score = np.where(np.isfinite(score), score, 0.0)
    return score
