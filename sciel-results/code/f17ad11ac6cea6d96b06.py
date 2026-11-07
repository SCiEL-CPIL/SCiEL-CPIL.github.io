"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization with deadline guard.
    # Score(x) = -mu(x) + kappa * sigma(x)
    # Higher score means evaluate next.
    
    # Deadline guard: ensure we return well before 330s wall-clock limit
    t_start = time.time()
    deadline = t_start + 280.0  # 50s safety margin
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Adaptive kappa: reduce exploration as we find better solutions
    # and when uncertainty is low relative to the function range
    mu_range = np.ptp(mu) if len(mu) > 0 else 1.0
    sigma_mean = np.mean(sigma) if len(sigma) > 0 else 1.0
    
    # Normalize f_best relative to typical Ackley values (range ~0-10)
    # As f_best approaches 0, we should exploit more
    f_best_norm = np.clip(f_best / 10.0, 0.0, 1.0)
    
    # When f_best is close to optimum, reduce exploration
    exploit_factor = 1.0 - 0.5 * (1.0 - f_best_norm)
    
    # Base kappa between 0.5 and 1.5, scaled by exploit factor
    kappa = (0.5 + 1.0 * f_best_norm) * exploit_factor
    
    # If uncertainty is very low overall, increase exploration slightly
    if sigma_mean < 1e-3 and mu_range > 1e-3:
        kappa *= 1.2
    
    # Compute base LCB score
    scores = -mu + kappa * sigma
    
    # Check deadline after base computation
    if time.time() > deadline:
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Refine: dampen extreme uncertainty to focus on promising uncertain regions
    # Points with very high sigma relative to the range are likely in unexplored
    # or noisy regions; we want to explore but not waste evaluations on pure noise
    sigma_max = np.max(sigma) if len(sigma) > 0 else 1.0
    if sigma_max > 1e-10:
        sigma_norm = sigma / sigma_max
        # Dampen scores where normalized sigma is very high (>0.95)
        # This keeps exploration but avoids pure noise regions
        damp_mask = sigma_norm > 0.95
        if np.any(damp_mask):
            # Apply a gentle penalty proportional to how far above threshold
            excess = (sigma_norm[damp_mask] - 0.95) / 0.05
            scores[damp_mask] -= 0.1 * excess * sigma_max
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
