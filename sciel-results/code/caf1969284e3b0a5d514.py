# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition for Bayesian optimization (minimization).
    
    Structural change: Uses Expected Improvement (EI) instead of Probability of Improvement (PoI).
    EI captures the magnitude of expected regret reduction, which is more informative for 
    minimizing simple regret than just the probability of improvement.
    
    score(x) = EI(x) / cost(x) + beta * sigma(x) / cost(x)
    
    The cost normalization ensures we maximize improvement per unit budget spent.
    The exploration bonus is dynamically scaled:
    - Early in optimization (high f_best), beta is larger to encourage exploration.
    - Late in optimization (low f_best), beta is smaller to focus on exploitation.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    t_start = time.time()
    deadline = 280.0  # Safety margin before the 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Numerical guards: ensure sigma and cost are positive and finite
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-10), sigma, 1e-6)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-10), cost, 1.0)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)

    # Compute Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    improvement = f_best - mu_safe
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    z = np.clip(z, -40.0, 40.0)
    
    # Standard normal CDF and PDF
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # EI calculation
    ei = improvement * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical guard)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Structural change: use linear cost penalty + uncertainty bonus instead of cost division.
    # Division by cost overly penalizes expensive regions; linear subtraction allows
    # high-EI expensive points to compete when improvement outweighs cost.
    #
    # score = EI(x) - penalty * cost(x) + bonus * sigma(x)
    #
    # Penalty scales with f_best: early on (high f_best), we're less sensitive to cost
    # because potential improvements are large. Late on (low f_best), we're more
    # sensitive to cost because remaining improvements are small.
    #
    # The uncertainty bonus (sigma) encourages exploration in uncertain regions,
    # which is especially valuable in the small-budget regime (~25 evaluations).
    
    # Progress: 0 when f_best is high, 1 when f_best is low (near minimum)
    if f_best <= 0:
        progress = 1.0
    else:
        # Scale relative to typical initial f_best range for this function
        progress = np.clip(1.0 - f_best / 1.5, 0.0, 1.0)
    
    # Cost penalty: decreases as we converge (we can afford to be more selective)
    # Early: penalty ~0.3 (cost matters less when improvements are large)
    # Late: penalty ~1.0 (cost matters more when improvements are small)
    penalty = 0.3 + 0.7 * progress
    
    # Uncertainty bonus: encourages exploration
    # Higher early (more exploration needed), lower late (focus on exploitation)
    # But never zero - always some exploration to avoid local minima
    bonus = 0.5 * (1.0 - 0.5 * progress)
    
    # Final score: EI - penalty * cost + bonus * sigma
    score = ei - penalty * cost_safe + bonus * sigma_safe
    
    # Ensure all scores are finite (higher = evaluate next)
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    if (time.time() - t_start) > deadline:
        return score

    return score
# EVOLVE-BLOCK-END
