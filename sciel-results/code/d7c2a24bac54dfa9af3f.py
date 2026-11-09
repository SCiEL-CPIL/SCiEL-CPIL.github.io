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
    
    # Adaptive beta: 
    # The insight from the best scoring programs (1.5379) suggests that exploration is key.
    # The current best (1.0933) uses beta=1.5.
    # Let's try to be slightly more aggressive or tuned.
    # However, the observed changes show that p23 (beta=1.5) scored 1.0933, while its parent p17 (beta=1.0) scored 1.5379.
    # Wait, the JSON says:
    # parent_observation score: 1.5378... (p17)
    # child_observation score: 1.0933... (p23)
    # So p17 (beta=1.0) was BETTER than p23 (beta=1.5).
    # This contradicts the "Knowledge from this run" text which says "Pure UCB with beta=1.5 ... Measured: 0.9765 -> 1.093".
    # The "Knowledge" text might be misleading or referring to a different comparison.
    # The JSON is the ground truth for recent history.
    # p17 had beta=1.0 (from inspiration 2 code which is likely p17 or similar).
    # p23 had beta=1.5 (current program).
    # Score 1.5379 > 1.0933.
    # Therefore, beta=1.0 is better than beta=1.5 in this context.
    
    # Let's revert to beta=1.0 or even slightly lower to prioritize exploitation more, 
    # but keep the UCB structure.
    
    beta = 1.0
    scores = -mu + beta * sigma
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
