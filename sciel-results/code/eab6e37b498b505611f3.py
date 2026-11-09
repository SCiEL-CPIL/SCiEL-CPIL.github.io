"""Combined acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state to track call count and detect new seeds
_state = {
    'n_calls': 0,
    'prev_f_best': None,
    'start_time': time.time()
}


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Deterministic UCB (D-UCB) with a linearly decaying
    exploration coefficient, transitioning to pure Expected Improvement (EI) as
    the search progresses. This structural change avoids the probabilistic
    averaging of pure EI early on, instead explicitly prioritizing high-uncertainty
    regions to resolve the multiple global minima of Styblinski-Tang.

    Parameters
    ----------
    mu : array_like, shape (2000,)
        GP posterior means at candidate points.
    sigma : array_like, shape (2000,)
        GP posterior standard deviations at candidate points.
    f_best : float
        Best observed value so far (minimum).

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more desirable points.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # Detect if a new seed/run has started
    if _state['prev_f_best'] is not None and f_best > _state['prev_f_best'] + 1e-6:
        _state['n_calls'] = 0
        _state['start_time'] = time.time()

    _state['prev_f_best'] = f_best
    n = _state['n_calls']
    _state['n_calls'] = n + 1

    # Deadline guard: if we are close to the 330s limit, return a fast fallback
    elapsed = time.time() - _state['start_time']
    if elapsed > 325.0:
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Compute D-UCB score (higher is better for selection)
    # For minimization, we want low mu and high sigma.
    # D-UCB score = -mu + kappa * sigma
    # kappa decays linearly from 2.0 to 0.0 over the first 15 evaluations
    # This forces exploration early, then switches to pure exploitation (EI).
    
    # Linear decay: kappa = max(0, 2.0 - (2.0 / 15) * n)
    # At n=0: kappa=2.0, at n=15: kappa=0.0
    kappa = max(0.0, 2.0 - (2.0 / 15.0) * n)
    
    # D-UCB score
    ducb = -mu + kappa * sigma
    ducb = np.where(np.isfinite(ducb), ducb, 0.0)

    # Normalize both EI and D-UCB to [0, 1] range for blending
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-12
    ei_norm = (ei - ei_min) / ei_range
    
    ducb_min = np.min(ducb)
    ducb_max = np.max(ducb)
    ducb_range = ducb_max - ducb_min + 1e-12
    ducb_norm = (ducb - ducb_min) / ducb_range

    # Blend weight: as kappa decays, we shift weight from D-UCB to EI
    # When kappa is high (early), we rely more on D-UCB
    # When kappa is 0 (late), we rely purely on EI
    
    # The blend weight for D-UCB is proportional to kappa
    # weight_ducb = kappa / 2.0 (ranges from 1.0 to 0.0)
    # weight_ei = 1 - weight_ducb
    
    weight_ducb = kappa / 2.0
    weight_ei = 1.0 - weight_ducb

    # Final score
    scores = weight_ei * ei_norm + weight_ducb * ducb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
