"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses a deterministic Lower Confidence Bound (LCB) strategy for minimization
    with a high exploration coefficient to aggressively target regions of high
    posterior uncertainty, critical for escaping local optima in the multimodal
    Hartmann 6D landscape within a tight 30-evaluation budget.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Safety margin: If the total time exceeds 280s, we return a simplified score
    # to ensure we don't risk a timeout in the evaluator's loop.
    if time.time() - _start_time > 280:
        # Fall back to a pure exploitation strategy (minimize mu)
        # to finish quickly.
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Ensure sigma is non-negative to avoid invalid values
    sigma_safe = np.maximum(sigma, 0.0)
    
    # LCB for minimization:
    # For minimization, the lower confidence bound is mu - kappa * sigma.
    # We want to maximize the potential for finding a low value, so we
    # score by the negative of the LCB: -mu + kappa * sigma.
    # 
    # A high kappa (3.0) aggressively explores regions of high uncertainty,
    # which is critical for escaping local optima in the multimodal Hartmann 6D
    # landscape. This prioritizes global coverage over local exploitation.
    #
    # The score -mu + kappa*sigma favors points with:
    #   - Low posterior mean (exploit known good regions)
    #   - High posterior std (explore uncertain regions)
    #
    # kappa = 3.0 is chosen for aggressive exploration in 6D with only 30 evals.
    kappa = 3.0
    
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
