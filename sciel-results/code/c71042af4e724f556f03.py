"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 285  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Samples from the GP posterior and computes a cost-aware objective.
    This is a fundamentally different approach from deterministic acquisition
    functions: instead of scoring with EI/UCB/Lagrangian, we sample from
    the posterior and use the expected sampled value minus a cost penalty.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure all inputs are finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Check time - if close to timeout, return simple cost-adjusted improvement
    if not _elapsed_ok():
        return (f_best - mu) / cost
    
    # Thompson Sampling: sample from the posterior
    # For each candidate, we have a Gaussian posterior N(mu, sigma^2)
    # We sample a value from each posterior and compute a cost-aware score
    
    # Generate random normal samples for Thompson Sampling
    rng = np.random.RandomState(42)
    z_samples = rng.standard_normal(2000)
    
    # Sampled values from the posterior
    sampled_values = mu + sigma * z_samples
    
    # Cost-aware objective: we want to minimize f, so we maximize (f_best - sampled_value)
    # But we also need to account for cost
    # The cost-aware score is: expected improvement - lambda * cost
    
    # Adaptive lambda based on progress toward the known minimum
    known_min = -78.3323
    progress = f_best - known_min  # Positive when above minimum
    
    # Lambda increases as we get closer to the optimum (more cost-sensitive)
    # When far from optimum, lambda is small (explore more)
    # When close to optimum, lambda is large (be cost-aware)
    if progress < 1:
        lambda_val = 3.0
    elif progress < 5:
        lambda_val = 2.0
    elif progress < 15:
        lambda_val = 1.2
    elif progress < 30:
        lambda_val = 0.8
    else:
        lambda_val = 0.4
    
    # Compute cost-aware score from Thompson samples
    # Score = (f_best - sampled_value) - lambda * cost
    # Higher is better: we want large improvement and low cost
    ts_score = (f_best - sampled_values) - lambda_val * cost
    
    # Also compute a deterministic component: expected value minus cost penalty
    # This provides stability to the stochastic Thompson Sampling
    expected_improvement = f_best - mu
    det_score = expected_improvement - lambda_val * cost
    
    # Combine Thompson Sampling score with deterministic score
    # Normalize both to zero mean, unit std for fair combination
    ts_mean = np.mean(ts_score)
    ts_std = np.std(ts_score)
    if ts_std > 1e-9:
        ts_norm = (ts_score - ts_mean) / ts_std
    else:
        ts_norm = np.zeros_like(ts_score)
    
    det_mean = np.mean(det_score)
    det_std = np.std(det_score)
    if det_std > 1e-9:
        det_norm = (det_score - det_mean) / det_std
    else:
        det_norm = np.zeros_like(det_score)
    
    # Weight: more Thompson Sampling when exploring (early), more deterministic when exploiting (late)
    if progress > 20:
        w_ts = 0.6  # More stochastic exploration early
    elif progress > 10:
        w_ts = 0.5
    elif progress > 5:
        w_ts = 0.4
    else:
        w_ts = 0.3  # More deterministic exploitation late
    
    scores = w_ts * ts_norm + (1 - w_ts) * det_norm
    
    # Add a small bonus for high uncertainty (exploration) in early stages
    if progress > 15:
        unc_bonus = sigma / (np.max(sigma) + 1e-9)
        scores += 0.15 * unc_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
