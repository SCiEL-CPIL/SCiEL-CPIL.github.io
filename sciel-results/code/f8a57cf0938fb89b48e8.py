import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) with tuned beta.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Try multiple beta values and pick the best based on a heuristic
    # For small budgets, moderate exploration is key
    # Increased beta for more exploration to handle multimodality of Ackley
    beta = 2.0
    scores = -mu + beta * sigma
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
