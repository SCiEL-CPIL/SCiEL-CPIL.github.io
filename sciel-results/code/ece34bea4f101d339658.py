"""Cost-aware LCB acquisition for Bayesian optimization of Levy 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) with adaptive exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value so far.
    cost : ndarray, shape (2000,)
        Estimated cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Adaptive beta that decays as more budget is consumed.
    # cost is the cumulative cost array; use its mean as a proxy for budget consumed.
    # Early on (low mean cost), we want more exploration (higher beta).
    # Later (high mean cost), we want more exploitation (lower beta).
    # The total budget is 30, starting cost is ~5 (5 random points).
    # So budget consumed fraction is roughly mean(cost) / 30.
    
    # Actually, let's think about this differently.
    # The cost array represents the cost of evaluating each of the 2000 candidates.
    # The total cost budget is 30. We start with 5 random points.
    # The "budget consumed" isn't directly in the cost array - cost[i] is the cost
    # of evaluating candidate i, not the total cost so far.
    
    # Let me re-read the task: "While total cost is below 30, the evaluator scores
    # 2000 new uniform candidates with acquisition and evaluates the best one."
    
    # So cost[i] is the cost of evaluating candidate i. The total cost is the sum
    # of costs of all evaluated points. We don't have the total cost directly,
    # but we can infer it from the number of evaluations or use a heuristic.
    
    # Actually, looking at the seed program, it just uses cost directly as a
    # normalization factor. Let's use a similar approach but with LCB.
    
    # LCB: mu - beta * sigma
    # Cost-aware: divide by cost to prefer cheaper points
    
    # Adaptive beta: start high for exploration, decay over time.
    # We can estimate progress from f_best relative to the known minimum (0 for Levy).
    # Or we can use a simple heuristic based on the magnitude of f_best.
    
    # Let's use a beta that starts around 2.0 and decays to 0.5 as we get closer to optimum.
    # f_best for Levy 2D minimum is 0. Typical initial values are much larger.
    
    # Normalize f_best to get a progress estimate
    # Levy 2D at random points in [-10,10]^2 typically has values in [0, 100+]
    # Let's use a logarithmic decay
    
    f_best_val = float(f_best)
    
    # If f_best is very large, we're early; if close to 0, we're late
    # Use a soft scaling
    if f_best_val > 1e-6:
        progress = np.clip(np.log10(f_best_val) / np.log10(100.0), 0.0, 1.0)
    else:
        progress = 0.0
    
    # beta starts at 2.5 when progress=0 (early) and decays to 0.5 when progress=1 (late)
    beta = 0.5 + 2.0 * (1.0 - progress)
    
    # LCB score: lower is better, so we negate for "higher = better"
    lcb = mu - beta * sigma
    
    # Cost-aware: prefer lower cost
    # Use LCB / cost, but ensure finite values
    scores = lcb / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
