"""Acquisition function for Bayesian optimization of hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Keep a module-level start time for the deadline guard
_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Uses a combination of Expected Improvement (EI) and a small amount of
    exploration via the lower confidence bound (LCB) to balance exploitation
    and exploration. This structural change from pure EI to a hybrid EI+LCB
    approach should help escape local optima in the hartmann 3d landscape.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (in objective units, lower is better)
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (in objective units)
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates
    """
    # Ensure sigma is strictly positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute the z-score for expected improvement
    # Since we're minimizing, improvement is when mu < f_best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (LCB) for exploration
    # For minimization: LCB = mu - kappa * sigma
    # We want to maximize the acquisition, so we negate to get a "score"
    # A lower LCB value (more negative) is better for minimization
    # So score_lcb = -LCB = -mu + kappa * sigma
    kappa = 1.0  # Exploration parameter
    lcb_score = -mu + kappa * sigma_safe
    
    # Normalize both components to [0, 1] range for fair combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    lcb_min, lcb_max = np.min(lcb_score), np.max(lcb_score)
    
    ei_range = ei_max - ei_min + 1e-10
    lcb_range = lcb_max - lcb_min + 1e-10
    
    ei_norm = (ei - ei_min) / ei_range
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # Combine: weight EI more heavily since we have some data, but include
    # LCB to encourage exploration in uncertain regions
    # Use a dynamic weight that decreases as f_best improves (more exploitation later)
    # For hartmann 3d with min -3.86278, if f_best is close to optimum, exploit more
    # If f_best is far, explore more
    # Simple heuristic: weight based on relative improvement
    # Actually, let's use a fixed reasonable weight for stability
    w_ei = 0.7
    w_lcb = 0.3
    
    scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
