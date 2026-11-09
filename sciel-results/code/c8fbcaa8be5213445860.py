import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We want to find the point that minimizes (mu - beta * sigma)
    # The evaluator selects the candidate with the HIGHEST score.
    # So we return the negative of the UCB value: score = beta * sigma - mu
    
    # Beta controls the exploration-exploitation trade-off.
    # A value of 1.0 corresponds to roughly the 16th percentile of the posterior distribution.
    beta = 1.0
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    scores = beta * sigma - mu
    
    return scores
