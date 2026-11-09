"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: hybrid of Expected Improvement (toward true min 0)
    and a noise-informed exploration term, with a mild penalty for predicted
    values far above the current best.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (objective units).
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (objective units).
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher is more promising to evaluate next.
    """
    t0 = time.time()
    
    # Safety floor on sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # --- Expected Improvement toward the known global minimum (0) ---
    # Using 0 as the reference makes the acquisition aggressively target the
    # true optimum rather than just improving on the current best.
    target = 0.0
    z = (target - mu) / sigma_safe
    ei = (target - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # --- Exploration term: reward high uncertainty relative to the scale ---
    # Scale is chosen robustly: at least 1.0 so that when f_best is near 0,
    # exploration is not suppressed; also capped to avoid extreme values.
    scale = max(abs(f_best), 1.0)
    rel_sigma = sigma_safe / scale
    # Soft cap: points with sigma > 2*scale get diminishing extra reward
    expl = np.log1p(2.0 * rel_sigma)
    
    # --- Penalty for points predicted to be much worse than current best ---
    # This prevents the exploration term from dragging us to obviously bad regions.
    # Use a soft penalty: only kicks in when mu > f_best significantly.
    penalty = np.maximum(0.0, (mu - f_best) / (scale + 1e-10))
    penalty = np.minimum(penalty, 3.0)
    
    # --- Combine ---
    # Weight EI heavily (it directly estimates improvement toward 0),
    # add exploration to break ties and explore under-sampled areas,
    # subtract a small penalty to avoid obviously poor candidates.
    scores = 2.0 * ei + 1.0 * expl - 0.5 * penalty
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard (should rarely trigger for this lightweight computation)
    if time.time() - t0 > 320.0:
        pass  # scores already computed; just return
    
    return scores
# EVOLVE-BLOCK-END
