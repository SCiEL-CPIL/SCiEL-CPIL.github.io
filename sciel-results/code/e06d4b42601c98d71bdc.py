import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Structural change: Use Cost-Weighted Lower Confidence Bound (LCB)
    # instead of Cost-Weighted Expected Improvement (EI).
    # LCB encourages exploration by penalizing high uncertainty (sigma)
    # in the minimization context (mu - beta * sigma).
    # We maximize the score: (beta * sigma - mu) / cost.
    
    # Exploration parameter beta
    # Tuned for small budget BO on Hartmann 3D
    beta = 2.0
    
    # Calculate scores
    # mu, sigma, cost are arrays of shape (2000,)
    # f_best is a scalar
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-6)
    
    scores = (beta * sigma - mu) / cost
    
    # Handle any numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
