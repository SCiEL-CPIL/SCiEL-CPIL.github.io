"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is non-negative and handle numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, improvement is f_best - f(x)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration Bonus
    # Use f_best as a proxy for exploration intensity.
    # Early on, f_best is large, so we explore more.
    # Late on, f_best is small, so we exploit more.
    # Scale sigma by f_best.
    # If f_best is 0, no exploration.
    # If f_best is large, strong exploration.
    
    # To avoid excessive exploration if f_best is very large (e.g. 100),
    # we might cap it or use a log scale.
    # However, Powell 4D values are usually moderate.
    
    # Let's use a dynamic lambda.
    # lambda = f_best / (1 + f_best) ? No, that's bounded.
    # Let's just use f_best directly, but scaled.
    
    # The previous successful run used "sigma/cost" added to "EI/cost".
    # That is equivalent to lambda=1.
    # Let's try lambda = 1.0 + f_best.
    # This increases exploration as f_best is high.
    
    exploration_bonus = (1.0 + f_best) * sigma
    
    # Combined Score
    score = (ei + exploration_bonus) / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
