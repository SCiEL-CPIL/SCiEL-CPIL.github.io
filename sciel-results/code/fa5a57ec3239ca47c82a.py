import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu (numpy array): Mean of the GP posterior, shape (N,)
    sigma (numpy array): Standard deviation of the GP posterior, shape (N,)
    f_best (float): The lowest observed value so far.
    
    Returns:
    numpy array: Scores for each candidate, shape (N,). Higher scores are preferred.
    """
    # Explicit deadline guard
    start_time = time.time()
    # Allow up to 290 seconds to be safe, though this operation is ~microseconds
    deadline = start_time + 290.0

    # Exploration parameter beta
    # Based on measured knowledge: beta=2.5 performed significantly better than EI
    beta = 2.5

    # Ensure sigma is strictly positive to avoid any potential numerical issues, 
    # although UCB formula beta*sigma - mu is stable with sigma=0.
    # We use a small epsilon to be safe against negative sigmas due to numerical noise.
    sigma_safe = np.maximum(sigma, 1e-12)

    # For minimization, the Lower Confidence Bound (LCB) is mu - beta * sigma.
    # We want to find the minimum of LCB.
    # The evaluator picks the candidate with the HIGHEST score.
    # Therefore, we return the negative of the LCB: -(mu - beta * sigma) = beta * sigma - mu.
    scores = beta * sigma_safe - mu

    # Check if we are approaching the deadline (highly unlikely for this size)
    if time.time() > deadline:
        # If we somehow ran out of time (e.g. extreme memory allocation failure),
        # we return the current best effort. Since the computation is already done,
        # we just return it.
        return scores

    return scores
