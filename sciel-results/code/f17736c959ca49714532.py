"""Model-and-optimize acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SECONDS = 300.0
_SAFETY_MARGIN = 20.0

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining EI with Monte Carlo probability of being
    the global minimum, plus adaptive exploration weighting.
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Check deadline guard
    if _elapsed() > _DEADLINE_SECONDS - _SAFETY_MARGIN:
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    n = len(mu)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Monte Carlo estimate of probability of being the global minimum
    # Sample from the GP posterior at each candidate: f_i ~ N(mu_i, sigma_i^2)
    # P(candidate i is the minimum) = mean over samples of [f_i == min(f)]
    #
    # To keep this fast, use a limited number of samples.
    # For 2000 candidates, even 100 samples means 200K operations per sample set.
    # We'll use 200 samples as a balance between accuracy and speed.
    
    # Time budget check: if we've already spent significant time, reduce samples
    elapsed_now = _elapsed()
    time_remaining = _DEADLINE_SECONDS - _SAFETY_MARGIN - elapsed_now
    if time_remaining < 30:
        n_samples = 50
    elif time_remaining < 60:
        n_samples = 100
    else:
        n_samples = 200
    
    # Generate samples: shape (n_samples, n)
    # f_samples[j, i] ~ N(mu[i], sigma[i]^2)
    rng = np.random.default_rng(42)
    samples = mu[None, :] + sigma_safe[None, :] * rng.standard_normal((n_samples, n))
    
    # Find the minimum index in each sample
    min_indices = np.argmin(samples, axis=1)
    
    # Count how often each candidate is the minimum
    min_counts = np.bincount(min_indices, minlength=n).astype(np.float64)
    p_global_min = min_counts / n_samples
    
    # Also compute: expected value of being the minimum (for a stronger signal)
    # E[f | f is min] weighted by probability
    # Simpler: use the probability directly as a bonus
    
    # Adaptive exploration: blend between exploitation (EI) and exploration
    # based on how far we are from convergence
    
    # Measure convergence: if EI values are all small relative to sigma,
    # we're in a more exploratory regime
    sigma_mean = np.mean(sigma_safe)
    ei_max = np.max(ei) if n > 0 else 0.0
    
    # Convergence indicator: high when EI is small relative to uncertainty
    if sigma_mean > 1e-10:
        convergence_ratio = ei_max / sigma_mean
    else:
        convergence_ratio = 1.0
    
    # When convergence_ratio is low, we should explore more
    # Use a smooth weight for the global-minimum bonus
    # Higher weight when we're less converged (more uncertainty)
    w_global = 0.5 + 0.5 * np.exp(-convergence_ratio)
    
    # Normalize p_global_min to be on a comparable scale to EI
    # Scale it so its max contribution is reasonable
    p_max = np.max(p_global_min) if n > 0 else 1.0
    if p_max > 1e-15:
        p_global_norm = p_global_min / p_max
    else:
        p_global_norm = np.zeros_like(p_global_min)
    
    # Additional exploration component: reward high sigma in regions
    # where the mean is not obviously bad
    sigma_max = np.max(sigma_safe) if n > 0 else 1.0
    if sigma_max > 1e-15:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    mu_min = np.min(mu)
    mu_range = np.max(mu) - mu_min + 1e-15
    mu_norm = (mu - mu_min) / mu_range  # 0 at best mean, 1 at worst
    
    # Exploration score: high sigma, low mean
    exploration_score = sigma_norm * (1.0 - 0.3 * mu_norm)
    
    # Weight for exploration: higher when uncertainty is high
    w_expl = 0.15 * (1.0 - np.tanh(convergence_ratio))
    
    # Combine:
    # score = EI + w_global * (scaled global min probability) + w_expl * exploration
    # Scale the global min probability to be comparable to EI
    # EI is in objective units; p_global is in [0,1]
    # Scale by the typical EI magnitude
    ei_scale = np.mean(ei[ei > 0]) if np.any(ei > 0) else sigma_mean
    if ei_scale < 1e-10:
        ei_scale = sigma_mean + 1e-10
    
    scores = ei + w_global * ei_scale * p_global_norm + w_expl * ei_scale * exploration_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
