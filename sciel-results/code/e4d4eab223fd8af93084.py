"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB with time-decaying exploration.
    # f_best is scalar, mu, sigma, cost are arrays of shape (2000,)
    
    # Ensure sigma is positive to avoid division issues, though GP sigma should be > 0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate exploration parameter beta based on total cost used so far.
    # The problem states total cost is below 30.
    # We assume 'cost' passed here is the cost of the *candidate* points, not the cumulative cost.
    # However, the signature is acquisition(mu, sigma, f_best, cost).
    # The prompt says: "While total cost is below 30... evaluator scores 2000 new uniform candidates with acquisition".
    # It does NOT pass the *cumulative* cost to the acquisition function, only the *cost* of the candidates (shape 2000).
    # Wait, looking at the seed program: `return ... / cost`. It divides by the candidate cost.
    # The prompt says: "mu, sigma and cost have shape (2000,)."
    # So we do NOT have access to the *total* accumulated cost in the function signature.
    # We only have the cost of the *current* candidates.
    
    # This limits our ability to do time-decaying beta based on *total* budget used.
    # We can only use a static beta or one derived from f_best?
    # If we cannot track time, we must use a static or stateless heuristic.
    # However, f_best is a proxy for progress. As f_best decreases, we are closer to optimum.
    # But f_best can get stuck.
    
    # Alternative: Use a static UCB with a tuned beta.
    # EI is essentially a specific form of UCB with a non-linear, data-dependent beta.
    # A linear UCB with a fixed beta might be more robust if the beta is chosen well.
    
    # Let's try a hybrid: Use EI but modify it to be more exploratory.
    # Or stick to UCB with a fixed beta.
    # For Rosenbrock, beta=1.5 to 2.0 is often good for UCB in BO.
    
    # Let's try: Score = (f_best - mu + beta * sigma) / cost
    # This is equivalent to maximizing (beta * sigma - mu) / cost.
    # This is "Expected Improvement" only if beta is derived from the distribution.
    # A fixed beta UCB is a different strategy.
    
    beta = 1.8  # Tuned for exploration in 2D Rosenbrock
    
    # UCB value for minimization is mu - beta*sigma.
    # We want to maximize the "desirability", which is - (mu - beta*sigma) = beta*sigma - mu.
    # We also want to account for cost.
    # Dividing by cost penalizes expensive points.
    
    scores = (beta * sigma_safe - mu) / cost
    
    # To ensure finite scores and avoid extreme outliers from very low cost (which shouldn't happen as min cost is 1.0)
    # or very high sigma.
    # The problem states cost = 1 + mean(x scaled). Min cost 1, Max cost 2.
    # So division is safe.
    
    return scores
# EVOLVE-BLOCK-END
