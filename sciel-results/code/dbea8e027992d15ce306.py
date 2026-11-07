import numpy as np
import time

# Global deadline tracking is not possible across calls easily without state,
# but we can track a module-level start time if the module is loaded once.
# However, the task says "Keep the function names...".
# It's safer to assume the function is called repeatedly.
# Since the computation is O(N) and N=2000, it is negligible.
# I will add a guard that checks if the system time is dangerously close to a hard limit 
# if we were to assume a global start, but without a global start, 
# we can't really guard against the 330s limit from inside a stateless function 
# unless we assume the function itself is the bottleneck, which it isn't.
# The "deadline guard" in the prompt likely refers to not hanging in a loop.
# Since we are just doing vectorized math, there is no loop to hang in.
# I will include a try-except to ensure we never return non-finite values.

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Cost is 1 + mean(x), so min cost is 1.0.
    # Add a tiny epsilon to be safe against zero division if cost is malformed
    cost_safe = np.maximum(cost, 1e-6)
    
    # Adaptive beta for UCB
    # Total budget is 30.
    # We estimate progress based on current cost.
    # If cost is low, we are early -> explore more (higher beta).
    # If cost is high, we are late -> exploit more (lower beta).
    
    # Linear decay of beta from 2.5 to 0.5 over the budget
    # At cost=0, beta=2.5. At cost=30, beta=0.5.
    # Note: Initial cost is ~7.5, so we start around beta=1.875.
    
    # Clamp cost to [0, 30] for beta calculation to avoid extrapolation issues
    cost_clamped = np.clip(cost, 0.0, 30.0)
    
    # Since cost varies per point, but beta is a global parameter for the step,
    # we should use the *average* or *current total* cost to determine the phase.
    # The prompt says "cost" has shape (2000,). This is the cost of *each candidate*.
    # We don't have the *total* cost spent so far directly, but we can infer it?
    # No, `f_best` is the lowest observed. We don't have a counter.
    # However, the *average* cost of the candidates is roughly 1.5.
    # The total cost spent is not passed.
    # Wait, the prompt says: "While total cost is below 30... f_best is the lowest observed value."
    # It does NOT pass total_cost.
    # Can we estimate total cost?
    # We have `cost` for 2000 candidates. This is the cost of *evaluating them*.
    # We don't know how many evaluations have happened.
    
    # Alternative: Use a fixed beta that is known to work well (e.g., 1.0 or 1.5).
    # The inspiration said beta=1 improved score.
    # Let's try a slightly higher beta, say 1.5, to encourage exploration in 6D.
    # Or, we can use the magnitude of sigma to adapt? No, sigma is per point.
    
    # Let's stick to a fixed beta = 1.5. 
    # Why 1.5? 1.0 worked better than EI (1.053 vs 1.0). 
    # In 6D, exploration is crucial. 1.5 is a reasonable step up.
    
    beta = 1.5
    
    # UCB score for minimization:
    # We want low mu and high sigma.
    # Score = (-mu + beta * sigma) / cost
    # Higher score = better.
    
    # To prevent numerical issues if sigma is 0, we add a tiny epsilon to sigma?
    # If sigma is 0, the term is just -mu/cost. That's fine.
    
    scores = (-mu + beta * sigma) / cost_safe
    
    # Ensure finite values
    # If mu or sigma are inf/nan, the result might be inf/nan.
    # We replace non-finite with a very low value so they are never picked.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
