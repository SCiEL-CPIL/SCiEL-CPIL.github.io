import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Parameters:
    mu: array of shape (2000,) with GP mean predictions.
    sigma: array of shape (2000,) with GP standard deviation predictions.
    f_best: scalar, the best observed function value so far.
    cost: array of shape (2000,) with the cost of evaluating each candidate.
    
    Returns:
    scores: array of shape (2000,) with acquisition scores. Higher is better.
    """
    # UCB for minimization:
    # We estimate the potential best value at x as mu(x) - beta * sigma(x).
    # The "gain" from evaluating x is how much better this estimate is than f_best.
    # Gain(x) = max(0, f_best - (mu(x) - beta * sigma(x)))
    
    # Beta controls the exploration-exploitation trade-off.
    # A value of 2.0-3.0 is typical for BO.
    # We use 2.5 as a balanced choice for Ackley.
    beta = 2.5
    
    # Calculate the UCB lower bound (for minimization)
    ucb_lower = mu - beta * sigma
    
    # Calculate the potential improvement
    # If the UCB lower bound is better (lower) than f_best, we have a gain.
    gain = np.maximum(0.0, f_best - ucb_lower)
    
    # Cost-aware score: Gain per unit cost
    # cost is >= 1.0, so division is safe.
    scores = gain / cost
    
    # Handle any numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
