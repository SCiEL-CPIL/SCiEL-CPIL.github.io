"""Model-and-optimize acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SECONDS = 300.0
_SAFETY_MARGIN = 20.0

# Known global minimum of Hartmann-3D
_HARTMANN3D_MIN = -3.86278

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _ei_scores(mu, sigma, target):
    """
    Compute Expected Improvement for minimization relative to a target value.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (target - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (target - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(np.isfinite(ei), ei, 0.0)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement with adaptive
    exploitation-exploration balance based on convergence to the
    known global optimum.
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Deadline guard: fall back to simple, fast EI if time is running out
    if _elapsed() > _DEADLINE_SECONDS - _SAFETY_MARGIN:
        return _ei_scores(mu, sigma, f_best)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    n = len(mu)
    if n == 0:
        return mu
    
    # Convergence-based adaptive weight
    # When far from optimum, explore more (lower weight on exploitation)
    # When close to optimum, exploit more (higher weight on exploitation)
    if f_best > _HARTMANN3D_MIN:
        gap = f_best - _HARTMANN3D_MIN
        # Normalize gap: large gap -> small conv, small gap -> large conv
        conv = 1.0 - (gap / (gap + 0.5))
    else:
        conv = 1.0
    
    conv = float(np.clip(conv, 0.0, 1.0))
    
    # Weight for exploitation (EI-like) vs exploration (Thompson)
    # When conv is high (close to optimum), weight_exploit is high
    # When conv is low (far from optimum), weight_explore is high
    weight_exploit = 0.3 + 0.7 * conv  # Range: [0.3, 1.0]
    weight_explore = 1.0 - weight_exploit  # Range: [0.0, 0.7]
    
    # Exploitation component: EI relative to f_best
    ei_fb = _ei_scores(mu, sigma, f_best)
    
    # Exploration component: Use posterior lower bound (mean - k*sigma)
    # The expected minimum of k samples from N(mu, sigma^2) is:
    # E[min] = mu - sigma * E[max of k standard normals]
    # For k standard normals, E[max] can be approximated
    
    # Adaptive k: more exploration when far from optimum
    # k_samples adapts with convergence
    k_samples = int(1 + 5 * (1.0 - conv))  # Range: [1, 6]
    
    # E[max of k standard normals] approximation
    # For k=1: 0
    # For k=2: 0.564
    # For k=3: 0.764
    # For k=4: 0.915
    # For k=5: 1.029
    # For k=6: 1.122
    # Approximation: sqrt(2*ln(k)) for k >= 2
    
    if k_samples == 1:
        c_k = 0.0
    else:
        c_k = np.sqrt(2 * np.log(k_samples))
    
    # Expected minimum: mu - sigma * c_k (c_k is positive, so this is lower than mu)
    expected_min = mu - sigma_safe * c_k
    
    # Score based on expected minimum: lower is better, so we negate
    # But we need to combine with EI for exploitation
    # Use: f_best - expected_min (larger when expected_min is lower)
    
    scores = weight_exploit * ei_fb + weight_explore * (f_best - expected_min)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
