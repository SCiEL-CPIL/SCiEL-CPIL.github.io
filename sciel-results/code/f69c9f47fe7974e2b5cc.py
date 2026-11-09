import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # z for Expected Regret: z = (mu - f_best) / sigma
    # We want E[max(0, f(x) - f_best)]
    # If mu < f_best, z is negative.
    # Let's use the standard formula for E[max(0, Y - c)] where Y ~ N(mu, sigma^2)
    # E[max(0, Y - c)] = (mu - c) * Phi((mu - c)/sigma) + sigma * phi((mu - c)/sigma)
    
    z = (mu - f_best) / sigma
    
    # Phi(z) is norm.cdf(z)
    # phi(z) is norm.pdf(z)
    
    er = (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # We want to minimize ER, so we maximize -ER / cost
    # Adding a small constant to avoid negative scores dominating if we want positive scores?
    # The prompt says "higher = evaluate next".
    # If all scores are negative, the "highest" is the one closest to 0.
    # This works.
    
    scores = -er / cost
    
    return scores
