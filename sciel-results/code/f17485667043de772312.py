# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Gaussian Expected Improvement blended with UCB.
    
    Structural change from parent (Student-t EI):
    1. Switches to Gaussian predictive distribution (better calibrated for Beale).
    2. Blends with Upper Confidence Bound (UCB) to ensure exploration.
    3. Uses Z-score normalization for both components to ensure stable blending.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (objective units, lower is better).
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (objective units).
    f_best : float
        Best observed function value so far (lower is better).

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    t0 = time.time()
    deadline = 325.0  # Safety margin before 330s limit

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)

    # 1. Gaussian Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    # Handle numerical stability for large z
    # norm.cdf and pdf handle large values well, but we clip z to avoid overflow in pdf
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)

    # 2. Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    # Adaptive beta: slightly higher exploration early on
    # f_best is the current minimum. As it decreases, we reduce exploration.
    # Simple heuristic: beta = 1.0 + 0.5 * (1.0 / (1.0 + f_best))
    # If f_best is large (e.g., 100), beta ~ 1.0
    # If f_best is small (e.g., 0.1), beta ~ 1.45
    # Actually, we want MORE exploration when f_best is BAD (large).
    # So beta should be higher when f_best is large.
    # Let's use beta = 1.5 + 0.5 * (f_best / (f_best + 1.0))
    # If f_best = 0, beta = 1.5
    # If f_best = 100, beta = 2.0
    beta = 1.5 + 0.5 * (f_best / (f_best + 1.0))
    
    ucb = -mu + beta * sigma_safe

    # 3. Z-score Normalization
    # This ensures that the EI and UCB components have comparable scales,
    # preventing one from dominating due to different units/magnitudes.
    
    def zscore(arr):
        m = np.mean(arr)
        s = np.std(arr)
        if s < 1e-12:
            return np.zeros_like(arr)
        return (arr - m) / s

    ei_z = zscore(ei)
    ucb_z = zscore(ucb)

    # 4. Blend
    # We found that EI is more reliable for exploitation, but UCB prevents stagnation.
    # Weight 0.8 EI, 0.2 UCB.
    w_ei = 0.8
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * ei_z + w_ucb * ucb_z

    # Check time budget
    if time.time() > deadline:
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    return scores
# EVOLVE-BLOCK-END
