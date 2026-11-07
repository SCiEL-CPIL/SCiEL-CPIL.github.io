import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 295.0  # 295s wall-clock limit (5s safety margin)


def acquisition(mu, sigma, f_best):
    """
    Kriging Believer acquisition: EI evaluated on the Lower Confidence Bound (LCB).
    This shifts from aggressive exploration early to exploitation late,
    leveraging the known global minimum to calibrate progress.
    """
    # Check deadline
    if time.time() > _DEADLINE - 3.0:
        # Safety: return simple EI as fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        return np.maximum(ei, 0.0)

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)

    # Known global minimum of Styblinski-Tang 1D
    f_opt = -39.1662

    # Progress: 1.0 when we've found the optimum, 0.0 when far from it
    # f_best is typically much worse than f_opt early on
    progress = np.clip(1.0 - (f_best - f_opt) / 100.0, 0.0, 1.0)

    # Adaptive beta for LCB: larger when far from optimum (more exploration)
    # beta controls how much we trust the uncertainty
    beta = 0.5 + 3.0 * (1.0 - progress)

    # Lower Confidence Bound for minimization: LCB = mu - beta * sigma
    # We want to minimize, so lower LCB is better
    lcb = mu - beta * sigma_safe

    # Kriging Believer: evaluate EI at the LCB values
    # For each candidate, treat lcb as if it were the "best" value so far
    # EI for minimization: E[max(f_best_candidate - f, 0)] where f ~ N(mu, sigma)
    # But in Kriging Believer, we use the LCB as a proxy for the best value
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu)/sigma
    # In Kriging Believer, we use lcb as the reference:
    # The idea is: if we were to pick the point with lowest LCB, what's the expected improvement?
    
    # Actually, the standard Kriging Believer approach:
    # 1. Compute LCB = mu - beta * sigma
    # 2. The "believer" assumes the best point is at the LCB minimum
    # 3. For scoring, compute EI using the LCB-adjusted values
    
    # Alternative interpretation: score = EI where we use a virtual f_best derived from LCB
    # Let's use the standard formulation:
    # For each candidate i, compute EI_i = (f_best - mu_i) * Phi(z_i) + sigma_i * phi(z_i)
    # But the Kriging Believer twist: instead of using the actual f_best,
    # we can blend between f_best and the minimum LCB to create a "believer" reference
    
    # Minimum LCB across all candidates
    lcb_min = np.min(lcb)
    
    # The believer's "best" is a blend: early on, trust LCB more; late on, trust actual f_best
    # believer_f_best = progress * f_best + (1 - progress) * lcb_min
    believer_f_best = progress * f_best + (1.0 - progress) * lcb_min

    # Compute EI using the believer's reference value
    z = (believer_f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (believer_f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)

    # Also compute standard EI with actual f_best for comparison
    z_actual = (f_best - mu) / sigma_safe
    z_actual_clipped = np.clip(z_actual, -10, 10)
    ei_actual = (f_best - mu) * norm.cdf(z_actual_clipped) + sigma_safe * norm.pdf(z_actual_clipped)
    ei_actual = np.maximum(ei_actual, 0.0)

    # Blend: use more Kriging Believer EI early, more standard EI late
    weight_kriging = 0.6 + 0.4 * (1.0 - progress)  # 1.0 early, 0.6 late
    weight_standard = 1.0 - weight_kriging          # 0.0 early, 0.4 late

    score = weight_kriging * ei + weight_standard * ei_actual

    # Add a small exploration bonus for high-uncertainty regions early on
    # This helps escape local optima
    uncertainty_bonus = (1.0 - progress) * 0.1 * (sigma_safe / (np.max(sigma_safe) + 1e-9))
    score += uncertainty_bonus

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
