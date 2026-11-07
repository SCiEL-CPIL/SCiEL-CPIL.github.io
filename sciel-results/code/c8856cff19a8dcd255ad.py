"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Prevent division by zero or negative costs
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-6)

    # Calculate expected improvement (EI)
    z = (f_best - mu) / sigma
    # Clip z to prevent numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Calculate CA-UCB score
    # The budget is 30. We start with 5 points.
    # Let's assume the total cost so far is not directly passed, but 'cost' is the cost of each candidate.
    # The problem states: "While total cost is below 30".
    # We don't know the exact total cost used so far, but we can estimate the "remaining" budget 
    # by assuming we are in the middle of the process. 
    # A common heuristic is to use the average cost of the population or a fixed remaining budget.
    # However, a more robust approach for CA-UCB is to use a dynamic beta.
    
    # Let's use a simple heuristic for the exploration constant beta.
    # Beta should decrease as we get closer to the budget limit.
    # Since we don't have the total cost used, we can approximate the "stage" of the optimization.
    # Early on, we want more exploration (higher beta). Later, exploitation (lower beta).
    
    # Heuristic: Assume the total cost used is proportional to the number of iterations.
    # But we don't have iteration count.
    # Let's look at the cost of the candidates.
    # If the mean cost is high, we are burning budget faster.
    
    # Alternative: Use a fixed beta that is tuned for the 30 cost budget.
    # Let's try a beta that scales with the standard deviation of the costs.
    # Or simply a constant beta that has been found to work well.
    
    # Let's try a dynamic beta based on the mean cost.
    # If mean cost is low, we can afford more exploration.
    # If mean cost is high, we should exploit.
    
    mean_cost = np.mean(cost)
    
    # Beta should be small enough to not overpower EI, but large enough to explore.
    # Let's try beta = 1.0 * (1.0 - mean_cost) 
    # This is a guess. Let's try a more standard CA-UCB form.
    
    # CA-UCB = mu + beta * sigma / cost
    # But we want to maximize this.
    # The EI is also a good measure of exploitation.
    # Let's blend EI and UCB.
    
    # Hybrid: Score = EI / cost + lambda * (mu + beta * sigma) / cost
    # This is getting complicated.
    
    # Let's stick to a pure CA-UCB with a well-chosen beta.
    # The knowledge from the run says: "Adaptive UCB-EI hybrid with cost-weighted blending shifts exploration→exploitation as median cost rises... Measured: 1 -> 1.633."
    
    # Let's implement that hybrid.
    # Score = (1 - alpha) * (EI / cost) + alpha * (mu + beta * sigma) / cost
    # Where alpha increases as the median cost rises.
    
    median_cost = np.median(cost)
    
    # Normalize median cost to [0, 1] assuming max cost is around 2 (1 + 1)
    # min cost is 1, max is 2.
    # So median_cost is in [1, 2].
    # Let's map [1, 2] to [0, 1] for alpha.
    alpha = np.clip((median_cost - 1.0) / 1.0, 0.0, 1.0)
    
    # Beta for UCB part. Let's use a constant beta = 2.0 for now.
    beta = 2.0
    
    ucb_term = (mu + beta * sigma) / cost
    ei_term = ei / cost
    
    # Blend
    score = (1.0 - alpha) * ei_term + alpha * ucb_term
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() > deadline:
         return score
        
    return score
# EVOLVE-BLOCK-END
