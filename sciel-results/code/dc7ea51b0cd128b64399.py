"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
# Set deadline to 320s to be safe against the 330s kill
_DEADLINE = _START_TIME + 320.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Lower Confidence Bound (LCB).
    
    The score is designed to maximize: -mu + beta * (sigma / cost)
    This penalizes high-cost regions in the exploration term (uncertainty bonus)
    while keeping the exploitation term (mean) invariant to cost, as per the directive.
    
    Parameters:
    -----------
    mu : numpy.ndarray
        GP mean predictions, shape (2000,).
    sigma : numpy.ndarray
        GP standard deviations, shape (2000,).
    f_best : float
        Best observed function value so far.
    cost : numpy.ndarray
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    scores : numpy.ndarray
        Acquisition scores, shape (2000,). Higher is better.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple fallback score if time is running out
        # Using a simple PI-like score as fallback for safety
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        cost_safe = np.maximum(cost, eps)
        z = (f_best - mu) / sigma_safe
        # Simple approximation of CDF using tanh for speed in emergency
        pi_scores = 0.5 * (1 + np.tanh(0.7978845608028654 * z))
        return np.nan_to_num(pi_scores / cost_safe, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure inputs are numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # LCB Formulation:
    # We want to minimize f(x).
    # Standard LCB for minimization: L(x) = mu(x) - beta * sigma(x)
    # We want to pick the x with the lowest L(x).
    # To convert to a maximization score (higher = better):
    # Score(x) = -L(x) = -mu(x) + beta * sigma(x)
    
    # Directive Modification:
    # "exploration term is scaled by the inverse cost (1/cost) ... while the mean term remains invariant"
    # Score(x) = -mu(x) + beta * (sigma(x) / cost(x))
    
    # Choose beta.
    # A common choice in BO is sqrt(2 * ln(n)) where n is the number of candidates or evaluations.
    # With 2000 candidates, sqrt(2 * ln(2000)) ~ 3.8.
    # However, since we are in a short budget regime and Rosenbrock is stiff,
    # we might need to balance this. 
    # Let's use a fixed beta of 1.5 to start, which is a moderate exploration rate.
    # If the mean term dominates too much, we might miss the valley.
    # If exploration dominates, we might waste budget.
    # Given the "cost-aware" nature, dividing sigma by cost (1 to 2) reduces the exploration term by 50% in the worst case.
    # So effective beta is roughly beta / 1.5 on average.
    # Let's try beta = 2.0 to compensate for the cost division.
    
    beta = 2.0
    
    # Calculate the score
    # The mean term is -mu
    # The exploration term is beta * sigma / cost
    scores = -mu + beta * (sigma_safe / cost_safe)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
