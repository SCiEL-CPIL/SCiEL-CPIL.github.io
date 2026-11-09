"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global counter to track the number of acquisition calls for time-decay
_acquisition_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Deterministic Upper Confidence Bound (UCB) acquisition function with
    time-decaying exploration coefficient for Bayesian optimization.
    
    Score = -mu + beta(t) * sigma, where beta(t) decreases over the 30-evaluation
    budget to shift from exploration to exploitation.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        The lowest observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        UCB scores; higher means evaluate next.
    """
    global _acquisition_call_count
    
    # Increment the call counter
    _acquisition_call_count += 1
    
    # Total expected evaluations is 30 (5 initial + 25 loop iterations? 
    # Task says "until 30 evaluations including the first 5". 
    # So the loop runs 25 times. The acquisition is called 25 times.
    # However, the counter might be reset or not. 
    # Let's assume the counter starts at 0 and goes up to 25.
    # To be safe, we cap the progress at 1.0.
    
    total_steps = 25  # Approximate number of acquisition calls in the loop
    progress = min(_acquisition_call_count / total_steps, 1.0)
    
    # Beta decays from 3.0 (high exploration) to 0.5 (exploitation)
    # Linear decay is simple and effective for small budgets
    beta = 3.0 - 2.5 * progress
    
    # UCB score for minimization:
    # We want to maximize (beta * sigma - mu)
    # This corresponds to minimizing (mu - beta * sigma)
    scores = beta * sigma - mu
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
