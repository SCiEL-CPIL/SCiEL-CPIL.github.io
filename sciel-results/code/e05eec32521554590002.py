import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Computes the Upper Confidence Bound (UCB) acquisition function
    with an adaptive beta parameter based on the current best value.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Handle empty inputs
    if len(mu) == 0:
        return np.array([])
        
    # Parameters for adaptive beta
    beta_min = 0.1
    beta_max = 3.0
    k = 0.5
    
    # f_best is the minimum observed value.
    # We use it to determine the balance between exploration and exploitation.
    # High f_best -> High beta (Explore)
    # Low f_best -> Low beta (Exploit)
    
    # Clip f_best to avoid issues if it's negative (though Ackley is >= 0)
    fb = max(float(f_best), 1e-6)
    
    # Calculate beta
    beta = beta_min + (beta_max - beta_min) * (1.0 - np.exp(-k * fb))
    
    # UCB Score: We want to minimize the function value.
    # The acquisition score should be higher for points we want to evaluate.
    # So we maximize: -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
