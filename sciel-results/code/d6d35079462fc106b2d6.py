"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Score(x) = -mu(x) + beta * sigma(x) - lambda * cost(x)
    
    -mu(x) encourages exploitation (lower function values).
    beta * sigma(x) encourages exploration (higher uncertainty).
    -lambda * cost(x) encourages low-cost evaluations.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 40s margin for 330s limit

    # Parameters
    beta = 1.8
    lambda_cost = 0.1

    # Ensure sigma is positive to avoid division issues in other contexts, 
    # though here we just multiply. Still, good practice.
    sigma_safe = np.maximum(sigma, 1e-6)

    # Calculate UCB score
    # Note: We are minimizing, so we want low mu. Hence -mu.
    # We want high sigma. Hence +beta*sigma.
    # We want low cost. Hence -lambda*cost.
    score = -mu + beta * sigma_safe - lambda_cost * cost

    # Handle any numerical issues
    score = np.where(np.isfinite(score), score, -np.inf)

    # Check deadline (computation is O(N), should be instant)
    if time.time() > deadline:
        pass # Return current score

    return score
# EVOLVE-BLOCK-END
