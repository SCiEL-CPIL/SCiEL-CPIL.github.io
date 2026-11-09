"""Cost-aware acquisition function for Bayesian optimization on rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_DEADLINE = 280.0
_start_time = time.time()


def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Quantile of the Posterior (QoP) acquisition for minimizing rastrigin 2d.
    
    Uses the p-th quantile of the posterior distribution: q = mu + sigma * Phi^{-1}(p).
    We minimize this quantile, so the score is -q / cost.
    This directly targets high-confidence low-value regions while penalizing expensive evaluations.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    cost_safe = np.maximum(cost, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Determine the quantile p.
    # For minimization, we want a low quantile.
    # p=0.5 is median. p=0.1 is optimistic (exploration).
    # Let's use an adaptive p that shifts from exploration to exploitation.
    # Early on (high f_best), use a lower p (more exploration/optimism).
    # Later (low f_best), use a higher p (more exploitation/conservatism).
    
    # f_best is typically in [0, 20] for Rastrigin on this domain.
    # Map f_best to a progress factor [0, 1] where 1 is best.
    # If f_best is 20, progress=0. If f_best is 0, progress=1.
    progress = 1.0 - np.clip(f_best / 20.0, 0.0, 1.0)
    
    # p ranges from 0.1 (exploration) to 0.4 (exploitation)
    # This keeps us looking at the lower tail but not just the absolute minimum.
    p = 0.1 + 0.3 * progress
    
    # Calculate the p-th quantile
    z_p = norm.ppf(p)
    quantile = mu + sigma_safe * z_p
    
    # Score: we want to minimize quantile, so maximize -quantile.
    # Normalize by cost.
    # Directive says "divided by the evaluation cost".
    scores = -quantile / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
