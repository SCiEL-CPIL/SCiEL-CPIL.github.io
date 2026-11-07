"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a robust variant of Expected Improvement (EI) with cost normalization,
    incorporating a small exploration bonus to avoid premature convergence
    and handle noisy/flat regions better.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Standard normal variable
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)

    # Cost-aware: divide by cost
    # We use a slightly different normalization to handle varying costs better.
    # The cost is 1 + mean(x_scaled), so it ranges from 1 to 2.
    # Dividing by cost directly is standard, but we can also consider
    # a penalty that is less aggressive for high costs if the EI is very high.
    
    # Primary score: EI / cost
    score = ei / cost_safe

    # Add a small exploration bonus based on uncertainty (sigma)
    # This helps escape local minima and explore high-uncertainty regions
    # The bonus is scaled down so it doesn't dominate the EI term
    # We use log(sigma) to penalize very low uncertainty more gently
    exploration_bonus = 0.1 * np.log(sigma_safe + 1e-9)
    
    # Combine EI/cost with a small exploration term
    # The exploration term is also divided by cost to maintain cost-awareness
    final_score = score + exploration_bonus / cost_safe

    # Ensure all scores are finite
    final_score = np.where(np.isfinite(final_score), final_score, -np.inf)

    return final_score
# EVOLVE-BLOCK-END
