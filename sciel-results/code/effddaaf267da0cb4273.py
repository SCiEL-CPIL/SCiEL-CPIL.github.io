import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    
    We want to minimize the function f(x).
    The LCB for minimization is mu(x) - beta * sigma(x).
    We want to pick the x that minimizes LCB(x).
    The acquisition function must return scores where HIGHER is better.
    Therefore, we return -LCB(x) = -mu(x) + beta * sigma(x).
    
    Args:
    mu: array of posterior means, shape (N,)
    sigma: array of posterior standard deviations, shape (N,)
    f_best: best function value observed so far (not used in LCB)
    
    Returns:
    scores: array of acquisition scores, shape (N,)
    """
    beta = 3.0
    # Ensure inputs are numpy arrays for safe operations
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    scores = -mu + beta * sigma
    return scores
