"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for deadline tracking
_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 330s limit
_DEADLINE = _start_time + 330.0 - _DEADLINE_MARGIN


def _check_deadline():
    """Return True if we should stop computing."""
    return (time.time() - _start_time) > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: scalar - lowest observed function value
    - cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if _check_deadline():
        # Return a simple cost-aware EI to finish quickly
        z = (f_best - mu) / (sigma + 1e-12)
        from scipy.special import ndtr, ndtrf
        # Use scipy.special for speed
        phi_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        cdf_z = ndtr(z)
        ei = (f_best - mu) * cdf_z + sigma * phi_z
        scores = ei / (cost + 1e-12)
        # Ensure finite
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Main acquisition computation
    # Cost-aware Expected Improvement with exploration bonus
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    from scipy.special import ndtr, ndtrf
    cdf_z = ndtr(z)
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-12)
    cost_ei = ei / cost_safe
    
    # Add a small UCB-style exploration bonus to encourage exploration
    # This helps escape local minima in the acquisition landscape
    # The bonus is scaled by uncertainty and inversely by cost
    exploration_weight = 0.1
    ucb_bonus = exploration_weight * sigma_safe / cost_safe
    
    # Combine: primarily cost-aware EI with a small exploration component
    scores = cost_ei + ucb_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative (EI can be negative if mu > f_best, but we want to focus on improvement)
    # Actually, negative EI means no expected improvement, so we keep it but it will be low priority
    
    return scores
# EVOLVE-BLOCK-END
