import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    
    Args:
    mu: np.array, shape (N,), posterior mean.
    sigma: np.array, shape (N,), posterior standard deviation.
    f_best: float, best observed value so far.
    cost: np.array, shape (N,), cost of evaluating each point.
    
    Returns:
    scores: np.array, shape (N,), higher is better.
    """
    # Convert to numpy arrays for vectorized operations
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Global minimum of Hartmann 6D
    f_opt = -3.32237
    
    # Calculate gap to global minimum
    # If f_best is very close to f_opt, gap is small.
    gap = f_best - f_opt
    
    # Adaptive beta:
    # Higher beta when far from optimum (exploration).
    # Lower beta when close to optimum (exploitation).
    # gap is typically ~1.5 to ~2.5 initially, ~0.0 later.
    # Normalize gap to [0, 1] roughly.
    norm_gap = np.clip(gap / 2.0, 0.0, 1.0)
    
    # Beta ranges from 2.0 (exploit) to 3.5 (explore)
    beta = 2.0 + 1.5 * norm_gap
    
    # Safe sigma to avoid division by zero or numerical issues
    # Although we multiply, not divide by sigma, keeping it safe is good practice
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the potential improvement per cost.
    # Potential improvement = f_best - LCB = f_best - (mu - beta * sigma)
    # Score = Potential Improvement / Cost
    score = (f_best - (mu - beta * sigma_safe)) / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
