"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track elapsed time for safety
_start_time = time.time()
_DEADLINE = 330.0  # Seconds
# We need to be safe. The evaluator kills at 330s.
# Each call is fast, but let's add a guard.

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses GP-UCB (Upper Confidence Bound) for maximization of a score
    that corresponds to minimization of the objective.
    
    For minimization of f, we want to find low f.
    The GP predicts mean mu and std sigma for the objective f.
    We want to maximize the probability of finding a low value.
    
    Standard UCB for minimization:
    We want to pick x that minimizes f(x).
    The "confidence bound" for the minimum is:
    LB = mu - beta * sigma
    
    We want to maximize the "expected improvement" or simply pick the point
    with the lowest Lower Bound (most likely to be low).
    So the score to maximize is: -LB = -mu + beta * sigma.
    
    Alternatively, using the standard formulation where we maximize an acquisition score:
    Score(x) = -mu(x) + beta * sigma(x)
    
    Why UCB over EI?
    In high dimensions (6D), EI can be overly conservative/greedy, leading to local optima.
    UCB with a fixed beta is a proven regret-minimizing strategy that balances exploration (sigma)
    and exploitation (mu) more linearly and robustly.
    
    Parameters:
    mu: array of shape (2000,) - GP posterior mean
    sigma: array of shape (2000,) - GP posterior std dev
    f_best: float - best observed value so far (lowest)
    
    Returns:
    scores: array of shape (2000,) - higher is better
    """
    # Check time budget
    # This check is cheap. If we are close to the deadline, we return zeros 
    # to fail gracefully (or rather, return a neutral score) rather than 
    # risking a timeout that invalidates the whole run.
    # However, the calculation below is O(N) and very fast for N=2000.
    # The timeout risk is unlikely for 2000 points unless the system is heavily loaded.
    # We will proceed with the calculation.
    
    # Ensure inputs are numpy arrays and 1D
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    # Safety: sigma should be non-negative. Clip small negative values due to numerical noise.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Hyperparameter beta for UCB.
    # A common choice in literature is beta ~ sqrt(2 * log(n)) or similar.
    # For a fixed budget of 30 evaluations, a constant beta is often sufficient.
    # beta=1.0 is a standard, balanced choice.
    # Higher beta -> more exploration. Lower beta -> more exploitation.
    # Given the 6D Hartmann function has multiple local minima,
    # a slightly higher exploration rate in early stages is good.
    # However, since we don't know the step number, a fixed beta is safest.
    # Let's try beta=1.0. If the current EI (score 1.0) is too greedy, UCB(1.0) might help.
    # Some sources suggest beta=2.0 or 3.0 for better exploration in high dims.
    # Let's stick to beta=1.0 as a robust baseline first, or maybe 1.5?
    # The prompt mentions "UCB with beta=1 balances exploration... Measured: 1 -> 1.044".
    # This suggests UCB beta=1 is an improvement.
    # Let's use beta=1.0.
    
    beta = 1.0
    
    # Calculate UCB score for minimization.
    # We want to maximize: -mu + beta * sigma
    # This represents the "lower confidence bound" of the function value,
    # but we invert it to a maximization score.
    # If mu is low and sigma is high, the score is high (good for exploration/exploitation).
    # If mu is high, the score is low (bad).
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

# EVOLVE-BLOCK-END
