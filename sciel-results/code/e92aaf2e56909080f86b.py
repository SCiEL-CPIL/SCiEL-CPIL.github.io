"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr, ndtri
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # wall-clock limit in seconds
_SAFETY_MARGIN = 15.0  # seconds before deadline to start returning


def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function returning 2000 finite scores.
    
    Uses a Cost-Adjusted Probability of Improvement (POI) where the improvement
    threshold is dynamically scaled by the remaining budget. As the cost budget
    depletes, the target demands larger relative improvements, preventing wasteful
    exploitation of marginal gains late in the run.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Estimate total cost spent so far.
    # The loop starts with 5 uniform random points (cost counted).
    # While total cost < 30, it evaluates the best of 2000 candidates.
    # We don't have the exact total cost, but we can infer it from the current
    # cost array and the fact that we are in a loop.
    # However, a simpler heuristic: the 'cost' parameter is the cost of evaluating
    # each candidate x. The total cost spent is not directly passed.
    # Let's use the number of evaluations inferred from the GP data? No, we don't have N.
    # The prompt says "While total cost is below 30".
    # We can approximate the "urgency" by assuming a typical progression.
    # Alternatively, we can use the spread of sigma or mu to gauge convergence.
    # But the directive specifically asks for: target = f_best - gamma * (30 - total_cost) / n_evals_remaining
    # Since we don't have total_cost or n_evals_remaining directly, we must estimate or use a proxy.
    
    # Proxy for remaining budget: 
    # If we assume the loop runs until cost ~ 30, and each step adds ~1.5 cost on average,
    # we've had ~ (30 - initial_cost) / 1.5 steps.
    # Initial cost is 5 points * ~1.5 = 7.5. Remaining ~ 22.5. Steps ~ 15.
    # This is too vague.
    
    # Let's look at the structure of the inspirations.
    # Inspiration 1 (score 1.0728) used simple POI / cost.
    # The current program (0.89) uses EI + UCB / cost.
    # The directive says: "Use a Cost-Adjusted Probability of Improvement (POI) where the improvement threshold is dynamically scaled by the remaining budget".
    
    # How to estimate remaining budget without explicit total_cost?
    # We can use the fact that f_best is decreasing. 
    # Or we can simply use a static gamma that mimics the effect.
    # However, the directive is specific. Let's assume we can estimate the number of evaluations.
    # We don't have the count. But we have mu and sigma for 2000 points.
    # Is there any other signal?
    # Maybe we can just use a fixed "urgency" factor that increases as f_best approaches the known minimum?
    # Known min is -3.32237.
    
    # Let's try a different interpretation: "remaining budget" can be approximated by how close we are to the theoretical optimum?
    # Or perhaps we can track the call count?
    # We can use a global variable to track the number of times acquisition is called.
    # Each call corresponds to one step in the loop.
    
    global _call_count
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
    acquisition._call_count += 1
    n_calls = acquisition._call_count
    
    # Estimate total cost.
    # Start: 5 points. Average cost per point in [0,1]^6 is 1 + 0.5 = 1.5.
    # Initial cost = 7.5.
    # Each subsequent step adds cost of the chosen point.
    # Average cost of chosen point might be lower if we bias towards cheap points.
    # Let's assume average cost per step is 1.3.
    # Total cost ~= 7.5 + 1.3 * n_calls.
    # Max cost 30.
    # Remaining budget = max(0, 30 - total_cost).
    
    initial_cost = 7.5
    avg_step_cost = 1.3
    estimated_total_cost = initial_cost + avg_step_cost * n_calls
    remaining_budget = max(0.0, 30.0 - estimated_total_cost)
    
    # n_evals_remaining:
    # If avg cost is 1.3, remaining steps = remaining_budget / 1.3.
    n_evals_remaining = max(1, int(remaining_budget / avg_step_cost))
    
    # Dynamic target:
    # target = f_best - gamma * (remaining_budget / n_evals_remaining)
    # This simplifies to f_best - gamma * avg_step_cost.
    # This is constant! That doesn't seem right.
    # The formula in the directive: f_best - gamma * (30 - total_cost) / n_evals_remaining
    # If n_evals_remaining is proportional to (30 - total_cost), the term is constant.
    # Perhaps n_evals_remaining is fixed? No, it's remaining.
    # Maybe the intent is that as budget depletes, we demand *larger* improvements?
    # If remaining budget is small, n_evals_remaining is small.
    # (30 - total_cost) is small.
    # The ratio is roughly constant if cost is linear in steps.
    
    # Let's re-read: "force the search to demand larger relative improvements as the budget depletes"
    # If we use EI, we are already demanding improvement.
    # If we use POI, we are demanding P(f < f_best).
    # To demand *larger* improvements, we should shift the target down.
    # target = f_best - delta.
    # As budget depletes, delta should increase?
    # If delta increases, we require f(x) < f_best - delta, which is a harder condition.
    # So POI = P(f < target) = Phi((target - mu)/sigma).
    # If target decreases, (target - mu) becomes more negative, POI decreases.
    # This penalizes points that are not significantly better than the current best.
    
    # How to scale delta?
    # Let delta = gamma * (1 - remaining_budget / max_budget).
    # As remaining_budget -> 0, delta -> gamma.
    # As remaining_budget -> max, delta -> 0.
    
    max_budget = 30.0 - initial_cost  # ~22.5
    budget_ratio = remaining_budget / max_budget if max_budget > 0 else 0
    # delta grows as budget depletes
    gamma = 0.5  # Scale factor, in objective units
    delta = gamma * (1.0 - budget_ratio)
    
    target = f_best - delta
    
    # Compute z-score for the new target
    z = (target - mu) / sigma
    
    # Probability of Improvement against the dynamic target
    poi = ndtr(z)
    
    # Cost normalization
    # Use linear cost normalization as in the best inspiration (1.0728)
    acq = poi / cost_safe
    
    # Ensure finite values
    acq = np.where(np.isfinite(acq), acq, 0.0)
    
    # Ensure non-negative
    acq = np.maximum(acq, 0.0)
    
    return acq
# EVOLVE-BLOCK-END
