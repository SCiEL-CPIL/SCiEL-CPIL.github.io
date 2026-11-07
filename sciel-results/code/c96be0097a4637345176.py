"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using improvement density per unit cost.
    Models the search as maximizing the density of the improvement distribution
    at the threshold (phi(z)) normalized by evaluation cost.
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviations.
        f_best: Scalar, the best observed function value so far.
        cost: Array of shape (N,) with evaluation costs for each candidate.
        
    Returns:
        Array of shape (N,) with scores. Higher score is better.
    """
    start_time = time.time()
    deadline = start_time + 290.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    f_best_val = float(f_best)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-8)

    # Compute z-scores: how many standard deviations below f_best the mean is
    # z = (f_best - mu) / sigma
    z = (f_best_val - mu) / sigma_safe
    
    # The improvement density at the threshold is phi(z), the standard normal PDF
    # This represents the density of the improvement distribution at f_best
    phi_z = norm.pdf(z)
    
    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    Phi_z = norm.cdf(z)
    ei = (f_best_val - mu) * Phi_z + sigma_safe * phi_z
    
    # Cost-aware scores
    # 1. EI per unit cost: balances expected improvement with evaluation cost
    ei_cost = ei / cost_safe
    
    # 2. Probability of Improvement (PI) per unit cost
    pi_cost = Phi_z / cost_safe
    
    # 3. Improvement density per unit cost
    density_cost = phi_z / cost_safe
    
    # 4. Exploration: uncertainty per unit cost
    explore_cost = sigma_safe / cost_safe
    
    # Adaptive weighting:
    # Early on (f_best is large relative to typical function values), favor exploration
    # Later (f_best is small), favor exploitation (EI and PI)
    
    # Use f_best as a proxy for how close we are to optimum
    # Griewank 2D has values in [0, ~0.5] typically, so scale accordingly
    f_best_scaled = np.clip(f_best_val / 0.5, 0.0, 1.0)
    
    # Weights: blend exploration and exploitation based on f_best
    # When f_best is high (far from optimum), explore more
    # When f_best is low (near optimum), exploit more
    
    w_ei = 0.6 * (1.0 - f_best_scaled) + 0.2
    w_pi = 0.3 * (1.0 - f_best_scaled) + 0.1
    w_density = 0.15 * (1.0 - f_best_scaled) + 0.05
    w_explore = 0.4 * f_best_scaled + 0.05
    
    # Normalize weights
    w_sum = w_ei + w_pi + w_density + w_explore
    w_ei /= w_sum
    w_pi /= w_sum
    w_density /= w_sum
    w_explore /= w_sum
    
    scores = w_ei * ei_cost + w_pi * pi_cost + w_density * density_cost + w_explore * explore_cost
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: simple EI per cost
        scores = ei_cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
