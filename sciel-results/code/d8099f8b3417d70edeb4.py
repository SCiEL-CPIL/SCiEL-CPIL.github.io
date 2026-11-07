"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global variables for deadline tracking
_START_TIME = None
_DEADLINE_LIMIT = 325.0  # 5 seconds margin before the 330s kill switch

def _get_elapsed_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (POI) per unit cost.
    
    This structural change shifts from Expected Improvement (EI) to POI, which can
    better balance exploration and exploitation in narrow valleys like Rosenbrock by
    focusing on the probability of finding a point better than the current best,
    normalized by the evaluation cost.
    
    Parameters:
    mu (np.ndarray): GP posterior means, shape (N,)
    sigma (np.ndarray): GP posterior standard deviations, shape (N,)
    f_best (float): Best observed function value so far.
    cost (np.ndarray): Evaluation costs for each candidate, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,). Higher scores indicate
                candidates that are more likely to reduce the regret
                per unit cost.
    """
    # Deadline guard: If we are close to the time limit, return a simple, fast heuristic
    # to ensure we return valid scores and don't crash/timeout.
    # The heuristic: Maximize -mu / cost (exploit known bests, prefer cheap).
    if _get_elapsed_time() > _DEADLINE_LIMIT:
        mu_arr = np.asarray(mu, dtype=np.float64)
        cost_arr = np.asarray(cost, dtype=np.float64)
        safe_scores = -mu_arr / np.maximum(cost_arr, 1e-8)
        safe_scores = np.where(np.isfinite(safe_scores), safe_scores, 0.0)
        return safe_scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-zero
    sigma_safe = np.maximum(sigma, 1e-8)
    # Ensure cost is non-zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # For minimization, if f(x) is Normal(mu, sigma),
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    poi = norm.cdf(z)
    
    # Expected Improvement magnitude (for exploitation signal)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * poi + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Adaptive weighting: blend POI and EI
    # When uncertainty is high (sigma large relative to gap), favor POI (exploration)
    # When uncertainty is low, favor EI (exploitation)
    # Use a smooth soft-min-like transition
    gap = np.maximum(f_best - mu, 1e-8)
    uncertainty_ratio = sigma_safe / gap
    
    # Weight for POI: higher when uncertain
    # Use a smooth sigmoid-like transition
    w_poi = 1.0 / (1.0 + np.exp(-2.0 * (uncertainty_ratio - 1.0)))
    w_ei = 1.0 - w_poi
    
    # Normalize EI to be comparable to POI (both in [0,1] roughly)
    # Scale EI by typical improvement magnitude
    ei_scale = np.maximum(ei, 1e-8)
    # Soft normalization: divide by (ei + typical value) to keep in (0,1)
    ei_norm = ei / (ei + gap * 0.5)
    ei_norm = np.clip(ei_norm, 0.0, 1.0)
    
    # Combined score: weighted blend of POI and normalized EI
    combined = w_poi * poi + w_ei * ei_norm
    
    # Cost-aware score: combined divided by cost
    scores = combined / cost_safe
    
    # Replace any non-finite values with 0.0 to ensure validity
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure scores are non-negative (higher = better)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
