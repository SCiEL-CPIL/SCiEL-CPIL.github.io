"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math


def acquisition(mu, sigma, f_best):
    """
    GP-UCB acquisition function for minimization.
    
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - lowest observed value so far
    
    Returns: array of shape (2000,) - acquisition scores (higher = better to evaluate)
    """
    # We need to estimate the number of evaluations t to set beta_t.
    # Since we don't have direct access to t, we can infer it from the context.
    # However, the signature only gives mu, sigma, f_best.
    # In the evaluator loop: starts with 5 points, then each step adds 1 point.
    # Total evaluations go from 5 to 30.
    # The function is called at each step to select the next point.
    # We don't have t directly. Let's think about how to handle this.
    
    # One approach: use a fixed reasonable beta, or try to infer t.
    # Since the problem states "for each seed... the evaluator starts from 5 uniform random points, 
    # then at each step draws 2000 fresh uniform candidates, computes the posterior there, 
    # calls acquisition and evaluates the candidate with the highest score, until 30 evaluations"
    
    # The number of evaluations so far when acquisition is called:
    # - First call: after 5 initial points, so t=5, we're choosing the 6th point
    # - Second call: after 6 points, t=6, choosing 7th
    # - ...
    # - Last call: after 29 points, t=29, choosing 30th
    
    # We don't have t in the signature. We need another way.
    # Let's use a heuristic: since we don't know t exactly, we can use a fixed beta 
    # that's reasonable for this range, or we can try to make beta adaptive.
    
    # Actually, looking more carefully, we can't determine t from mu, sigma, f_best alone.
    # Let's use a fixed beta that works well for the range t=5 to t=30.
    # beta_t = sqrt(log(t+1)) for t in [5, 30] gives beta in [sqrt(log(6)), sqrt(log(31))]
    # = [sqrt(1.79), sqrt(3.43)] = [1.34, 1.85]
    
    # A reasonable fixed beta around the middle: beta = 1.5
    # Or we could use beta = sqrt(log(15)) ≈ 1.58 as a representative value
    
    # Let's use a slightly higher beta for more exploration early on.
    # Since we can't vary with t, let's pick beta = 1.6 which is in the middle range.
    
    beta = 1.6
    
    # GP-UCB for minimization: maximize -(mu - beta * sigma) = -mu + beta * sigma
    # Or equivalently, we want to return scores where higher means better.
    # For minimization, the UCB is: ucb = -mu + beta * sigma
    # We return ucb as the score (higher = more promising for finding minimum)
    
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    ucb = -mu + beta * sigma_safe
    
    # Ensure all values are finite
    ucb = np.where(np.isfinite(ucb), ucb, -np.inf)
    
    return ucb
# EVOLVE-BLOCK-END
