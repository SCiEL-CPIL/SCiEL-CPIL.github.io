"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    
    Uses a combination of expected improvement and UCB, with cost-aware
    normalization and Pareto-frontier-inspired selection.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    t_start = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB exploration term with adaptive kappa
    # Use a moderate kappa that balances exploration
    kappa = 1.2
    ucb = f_best - mu + kappa * sigma
    
    # Use sqrt(cost) normalization to reduce over-penalization of high-cost points
    # This allows the algorithm to consider points that may contain the true minimum
    cost_factor = np.sqrt(cost)
    
    # Combine EI and UCB with a smooth blend
    # When uncertainty is high relative to improvement, weight UCB more
    improvement = np.maximum(f_best - mu, 0.0)
    uncertainty_ratio = sigma / (improvement + sigma + 1e-9)
    blend = np.clip(0.5 * uncertainty_ratio, 0.0, 1.0)
    
    combined = (1.0 - blend) * ei + blend * ucb
    
    # Normalize by cost factor
    scores = combined / cost_factor
    
    # Pareto-frontier-inspired selection:
    # Identify candidates that are non-dominated in (improvement, uncertainty, cost)
    # and boost their scores slightly to encourage diverse exploration
    
    # Compute normalized components for Pareto analysis
    imp_norm = (improvement - improvement.min()) / (improvement.ptp() + 1e-9)
    unc_norm = (sigma - sigma.min()) / (sigma.ptp() + 1e-9)
    cost_norm = (cost - cost.min()) / (cost.ptp() + 1e-9)
    
    # Non-dominated sort: a point is dominated if another point is better in all dimensions
    # For efficiency, use a simplified approach: compute a composite score and identify
    # points that are good in at least one dimension
    
    # Points with high improvement, high uncertainty, or low cost are valuable
    high_imp = imp_norm > 0.7
    high_unc = unc_norm > 0.7
    low_cost = cost_norm < 0.3
    
    # Boost points that are strong in at least one dimension
    boost_mask = high_imp | high_unc | low_cost
    scores[boost_mask] *= 1.15
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - t_start > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
