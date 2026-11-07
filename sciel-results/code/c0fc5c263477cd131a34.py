"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard and adaptive exploration
_bo_start_time = None
_bo_call_count = 0
_bo_total_evals = 30  # 5 initial + 25 more per the contract

def acquisition(mu, sigma, f_best):
    global _bo_start_time, _bo_call_count
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
    
    _bo_call_count += 1
    
    # Deadline Guard: 330s limit, use 305s as safety margin
    # If we are close to timeout, prioritize speed and exploitation
    # to ensure the loop finishes.
    if time.time() - _bo_start_time > 305:
        # Return pure exploitation score: lower mu is better
        mu = np.asarray(mu, dtype=np.float64)
        return -mu
    
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical instability
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Progress-based adaptive scheduling
    progress = min(_bo_call_count / _bo_total_evals, 1.0)
    
    # --- Component 1: Probability of Improvement (PI) ---
    # PI = Phi((f_best - mu) / sigma)
    # This is a deterministic, smooth measure of how likely a point is to beat f_best.
    # It naturally dampens pure noise exploration because points with high sigma
    # but mu > f_best will have low PI, unlike LCB which would score them high.
    # In the low-budget regime (30 evals), this avoids wasting evaluations on
    # high-uncertainty regions that are likely not actually good.
    Z_pi = (f_best - mu) / sigma_safe
    pi_score = norm.cdf(Z_pi)
    
    # --- Component 2: Expected Improvement (EI) ---
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # EI captures both the probability of improvement AND the magnitude of potential improvement.
    # It is more exploitative than PI in late stages because it rewards large gaps.
    Z_ei = (f_best - mu) / sigma_safe
    phi_Z = norm.pdf(Z_ei)
    Phi_Z = norm.cdf(Z_ei)
    ei_score = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Normalize both components to have comparable scales
    # Use robust normalization: center and scale by IQR or std
    def _robust_normalize(x):
        """Normalize array to [0, 1] range using min-max with clipping."""
        x = np.where(np.isfinite(x), x, 0.0)
        x_min = np.percentile(x, 1)
        x_max = np.percentile(x, 99)
        x_range = x_max - x_min
        if x_range < 1e-12:
            return np.zeros_like(x)
        return (x - x_min) / x_range
    
    pi_norm = _robust_normalize(pi_score)
    ei_norm = _robust_normalize(ei_score)
    
    # Blend: early = more PI (broad, smooth exploration), late = more EI (exploitation)
    # PI is smoother and less prone to picking noise than LCB, making it better
    # for the early low-budget phase where we need to cover the space efficiently.
    # EI takes over later to squeeze out the last bits of improvement.
    # weight_pi starts at 0.6 and decays to 0.2
    weight_pi = 0.6 * (1.0 - progress) + 0.2 * progress
    weight_ei = 1.0 - weight_pi
    
    scores = weight_pi * pi_norm + weight_ei * ei_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
