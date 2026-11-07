import numpy as np

def acquisition(mu, sigma, f_best):
    # f_max is the maximum possible value of x^2 on [-5, 5]
    f_max = 25.0
    
    # Calculate adaptive beta
    # As f_best decreases, we reduce exploration
    ratio = max(1e-6, f_best / f_max)
    beta = 2.0 * np.sqrt(ratio)
    
    # Lower Confidence Bound for minimization
    # We want to minimize (mu - beta * sigma)
    # The evaluator picks the MAX score
    # So we return the negative of the LCB value
    score = -mu + beta * sigma
    
    return score
