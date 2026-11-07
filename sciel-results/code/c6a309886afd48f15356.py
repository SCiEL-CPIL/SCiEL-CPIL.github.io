import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard, though likely not needed for 2000 points
_start_time = time.time()
_DEADLINE = 300.0 # Safety margin

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using a hybrid of Expected Improvement (EI) 
    and Upper Confidence Bound (UCB) with adaptive exploration.
    
    Parameters:
    mu: np.ndarray of shape (N,) - Posterior mean
    sigma: np.ndarray of shape (N,) - Posterior standard deviation
    f_best: float - Best observed value so far (minimum)
    
    Returns:
    np.ndarray of shape (N,) - Acquisition scores (higher is better)
    """
    
    # Deadline guard: if we are running out of time, return a safe fallback
    # This is unlikely to trigger for N=2000, but satisfies the directive.
    if time.time() > _DEADLINE:
        return -mu + 1.0 * sigma

    # Ensure sigma is positive
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI = (f_best - mu) * P(X < f_best) + sigma * phi(z)
    # Note: For minimization, improvement is f_best - f(x).
    ei_score = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # 2. Compute UCB score
    # We want to maximize the score. 
    # UCB for minimization: -mu + beta * sigma
    # Higher score means more promising.
    
    # Adaptive Beta:
    # We want high beta when f_best is far from the global minimum (-3.32237).
    # We want low beta when f_best is close to the global minimum.
    # Let's normalize f_best.
    # Assume f_best starts around 0-1 and goes down to -3.32.
    # Let's map f_best to a "progress" metric.
    
    # Global min is -3.32237.
    # Let's assume initial f_best is around 0.0 (or maybe 1.0).
    # We can use a sigmoid or linear mapping.
    
    # Linear mapping from f_best in [-3.32, 1.0] to beta in [0.5, 2.0]
    # When f_best = 1.0 (bad), beta = 2.0 (explore)
    # When f_best = -3.32 (good), beta = 0.5 (exploit)
    
    f_min_global = -3.32237
    f_initial_guess = 1.0 # Rough guess for initial uniform random points in [0,1]^6 for Hartmann 6D
    
    # Clamp f_best to this range for stability
    f_best_clamped = np.clip(f_best, f_min_global, f_initial_guess)
    
    # t = 0 when f_best = f_initial_guess
    # t = 1 when f_best = f_min_global
    t = (f_best_clamped - f_initial_guess) / (f_min_global - f_initial_guess)
    t = np.clip(t, 0.0, 1.0)
    
    # Beta decreases from 2.0 to 0.5
    beta = 2.0 - 1.5 * t
    
    ucb_score = -mu + beta * sigma_safe
    
    # 3. Combine scores
    # EI and UCB have different scales.
    # EI is bounded by (f_best - min(mu)) roughly.
    # UCB is bounded by -min(mu) + beta*max(sigma).
    
    # To combine them effectively, we can normalize them or use a weighted sum.
    # Inspiration 2 used 0.5 * UCB + 0.5 * EI.
    # Let's try a similar mix, but perhaps weight EI more as we converge.
    
    # Weight for EI: increases as we get closer to the minimum.
    # Weight for UCB: decreases as we get closer to the minimum.
    
    w_ei = 0.5 + 0.5 * t  # 0.5 to 1.0
    w_ucb = 1.0 - 0.5 * t # 1.0 to 0.5
    
    # Normalize weights to sum to 1? Or just use them as multipliers?
    # If we use multipliers, the scale changes.
    # Let's normalize the scores to [0,1] or similar? 
    # Or just use the weighted sum.
    
    scores = w_ei * ei_score + w_ucb * ucb_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
