import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize f(x).
    # LCB(x) = mu(x) - beta * sigma(x)
    # We maximize the negative of LCB per unit cost:
    # Score = (beta * sigma(x) - mu(x)) / cost(x)
    
    beta = 2.0
    
    # Ensure cost is not zero (it's >= 1, so safe)
    scores = (beta * sigma - mu) / cost
    
    # Handle any potential numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
