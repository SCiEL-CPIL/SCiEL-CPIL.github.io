"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or negative costs (though cost should be >= 1)
    # Cost is defined as 1 + mean(x_scaled), so min cost is 1.
    # We assume cost is valid.
    
    # Calculate remaining budget
    # Total budget is 30.
    # cost parameter in the signature represents the cost of evaluating each candidate.
    # The loop runs "While total cost is below 30".
    # We don't have the current total cost directly, but we can estimate remaining budget.
    # However, the prompt says "cost" has shape (2000,), implying it's the cost of each candidate.
    # The dynamic beta should depend on how much budget is left.
    # Since we don't have the current total spent, we can't perfectly compute remaining budget.
    # But we can use a heuristic: as we progress, we want to explore less.
    # Without explicit total cost, we can use a fixed but tuned beta, or estimate progress.
    # Let's look at the inspiration: "dynamic exploration term scaled by the remaining budget".
    # If we can't get remaining budget, we can approximate.
    # However, the function signature is fixed: acquisition(mu, sigma, f_best, cost).
    # We cannot access the loop's internal state.
    # Therefore, we must use a static or semi-static strategy that mimics the effect.
    # A common trick is to use a beta that decays or is tuned for the middle of the run.
    # Alternatively, we can use the spread of mu/sigma to adapt.
    
    # Let's try a Cost-Aware LCB:
    # LCB = mu - beta * sigma
    # But we want to normalize by cost?
    # The directive says: "cost-normalized Lower Confidence Bound (LCB) that explicitly penalizes high-uncertainty regions via a dynamic exploration term scaled by the remaining budget"
    
    # Since we can't access remaining budget, we will use a fixed beta that is effective for 8D.
    # In 8D, uncertainty is high. A standard LCB with beta ~ 1-2 often works well.
    # Let's try to incorporate cost into the penalty to discourage expensive points if they are uncertain.
    # Score = (mu - beta * sigma) / cost
    # This is similar to the current EI/cost but with LCB instead of EI.
    # EI can be very small if mu is close to f_best but sigma is large.
    # LCB explicitly subtracts uncertainty.
    
    # Let's refine the beta.
    # If we assume the run lasts ~20-30 evaluations (cost 30, avg cost ~1.5-2),
    # we have limited evaluations.
    # A slightly higher beta encourages exploration early.
    # Let's use beta = 1.5.
    
    beta = 1.5
    
    # Calculate LCB
    lcb = mu - beta * sigma
    
    # Normalize by cost
    # To avoid division by zero, we use a small epsilon
    scores = lcb / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    if time.time() > deadline:
         # If we are close to deadline, just return the current result
         # The calculation above is vectorized and fast, so this is unlikely to be hit
         pass
         
    return scores
# EVOLVE-BLOCK-END
