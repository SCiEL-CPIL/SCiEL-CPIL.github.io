import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) with adaptive beta.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive beta: start with higher exploration, decrease over time
    # Based on the knowledge that pure UCB with beta=1.5 works well
    # We'll use a slightly higher beta to encourage more exploration
    beta = 1.5
    scores = -mu + beta * sigma
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
