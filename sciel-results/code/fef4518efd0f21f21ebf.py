import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We maximize the score: -mu + beta * sigma.
    # This encourages exploration of regions with high uncertainty (high sigma)
    # and exploitation of regions with low predicted value (low mu).
    
    # Beta controls the exploration-exploitation trade-off.
    # A higher beta favors exploration.
    # Given the short budget and multimodal landscape, we choose a value 
    # that is more exploratory than the implicit balance of EI.
    beta = 2.5
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    return scores
