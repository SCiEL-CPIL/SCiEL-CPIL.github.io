import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 325.0

def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)

    # LCB for minimization: Score = mu - beta * sigma
    # We want to minimize f, so we prefer low mu.
    # We want to explore, so we prefer high sigma.
    # Maximizing (mu - beta * sigma) is equivalent to minimizing (mu - beta * sigma) if we invert?
    # No. 
    # Standard LCB for minimization: We minimize $L(x) = \mu(x) - \beta \sigma(x)$.
    # The acquisition function should return a score where HIGHER is better.
    # So we should return $A(x) = -L(x) = -\mu(x) + \beta \sigma(x)$?
    # Or simply return $L(x)$ and let the caller pick the MINIMUM?
    # The contract says: "Return 2000 finite scores, higher meaning evaluate next".
    # So if I compute LCB value $v = \mu - \beta \sigma$, I should return $-v$ to make lower LCB values correspond to higher scores.
    # Alternatively, I can just compute $Score = -\mu + \beta \sigma$.
    
    beta = 3.0
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Score to maximize: -mu + beta * sigma
    # This is equivalent to minimizing mu - beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
