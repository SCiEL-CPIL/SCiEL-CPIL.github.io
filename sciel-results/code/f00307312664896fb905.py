import numpy as np
import time

# Track global start time for the entire run
_global_start = time.time()
_TOTAL_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted UCB with adaptive exploration.
    
    The strategy uses a UCB-style score that balances exploitation (low mu) and
    exploration (high sigma), with the exploration bonus scaled by the square root
    of the remaining budget ratio to tighten exploration as the cost budget depletes.
    
    score = (-mu + beta * sigma * sqrt(remaining_budget / total_budget)) / cost
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Compute remaining budget ratio
    # Total budget is 30 (from the task description: "While total cost is below 30")
    # The cost array contains the cost of each candidate evaluation
    # We need to estimate remaining budget based on f_best and current costs
    
    # Estimate remaining budget:
    # The total budget is 30. The current cost spent is not directly available,
    # but we can estimate it from the cost array.
    # Actually, the cost array gives the cost of evaluating each of the 2000 candidates.
    # The actual cost spent so far is not directly given, but we can use the mean cost
    # as a proxy for the average cost per evaluation.
    
    # A simpler approach: use the mean of the cost array as the typical cost per evaluation
    # and estimate how many more evaluations we can afford.
    
    mean_cost = np.mean(cost)
    
    # Estimate remaining budget:
    # We know the total budget is 30. We don't know exactly how much has been spent,
    # but we can use a heuristic based on the improvement so far.
    # 
    # Alternative: use the ratio of current best to the range as a proxy for progress.
    # Ackley function minimum is 0, maximum is about 29.98 (at corners of [-5,5]^2).
    # So if f_best is close to 0, we're making good progress.
    
    # Let's use a different approach: track the number of evaluations made.
    # Since we don't have direct access to this, we'll use the cost array to estimate.
    
    # The cost array has shape (2000,), representing the cost of each candidate.
    # The actual cost spent is the sum of costs of all evaluated points.
    # We can estimate the remaining budget as:
    # remaining_budget = 30 - (estimated cost spent so far)
    
    # Since we don't know exactly how many points have been evaluated, we'll use
    # a conservative estimate. The problem states that runs start with 5 uniform random
    # points, and then the evaluator scores 2000 new candidates and evaluates the best one.
    # So each iteration adds one evaluation.
    
    # We can estimate the number of evaluations made by looking at the cost array.
    # The cost of each evaluation is 1 + mean(x scaled to [0,1]).
    # The minimum cost is 1 (when x is at the minimum of the scaled range).
    
    # Let's use a simpler heuristic: the remaining budget ratio is based on
    # how close f_best is to the minimum (0).
    
    # Ackley function: f(x) >= 0, with minimum 0 at x = (0, 0) in the original space.
    # In the shifted instances, the minimum is still 0.
    
    # Use f_best to estimate progress:
    # If f_best is close to 0, we're near the optimum, so we should exploit more.
    # If f_best is large, we're far from the optimum, so we should explore more.
    
    # A simple heuristic: remaining_budget_ratio = max(0, 1 - f_best / f_max)
    # where f_max is the maximum possible value of Ackley (approximately 29.98)
    
    f_max = 29.98  # Approximate maximum of Ackley function
    
    # Estimate remaining budget ratio
    if f_best <= 0:
        remaining_ratio = 1.0
    else:
        remaining_ratio = max(0.0, 1.0 - f_best / f_max)
    
    # Also consider the mean cost to adjust the ratio
    # Higher mean cost means we're spending more per evaluation, so we have less remaining budget
    cost_factor = mean_cost / 1.5  # Normalize by average cost (1 + 0.5 = 1.5 for uniform)
    remaining_ratio = remaining_ratio / (1.0 + 0.5 * cost_factor)
    
    # Clamp remaining_ratio to [0, 1]
    remaining_ratio = np.clip(remaining_ratio, 0.0, 1.0)
    
    # Adaptive beta:
    # Early on (f_best is poor), use larger beta for more exploration.
    # Later (f_best is good), use smaller beta for more exploitation.
    
    # Use f_best to determine beta:
    # If f_best > 10, we're early, use beta = 2.0
    # If f_best < 1, we're late, use beta = 0.5
    # Linear interpolation in between
    
    if f_best > 10:
        beta = 2.0
    elif f_best < 1:
        beta = 0.5
    else:
        # Linear interpolation between f_best = 1 and f_best = 10
        beta = 0.5 + (2.0 - 0.5) * (10 - f_best) / 9.0
    
    # Compute the UCB score with cost adjustment
    exploration_bonus = beta * sigma * np.sqrt(remaining_ratio)
    score = (-mu + exploration_bonus) / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > 290.0:
        return score
    
    return score
