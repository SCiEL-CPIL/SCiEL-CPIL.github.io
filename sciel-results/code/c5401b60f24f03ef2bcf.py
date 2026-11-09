import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 330.0  # Evaluator kills after 330s
_SAFETY_MARGIN = 10.0  # Safety margin

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (hartmann 3d).
    Blends Expected Improvement (EI), Probability of Improvement (PI),
    and a Thompson Sampling-like uncertainty bonus, all normalized and
    adjusted for evaluation cost.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Check deadline: if we're running out of time, return a simple fallback
    if _time_left() <= 0:
        # Fallback: exploit best (negative mu) divided by cost
        sigma_safe = np.maximum(np.asarray(sigma), 1e-9)
        cost_safe = np.maximum(np.asarray(cost), 1e-9)
        fallback = -np.asarray(mu) / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure inputs are properly shaped and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI and PI
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -50, 50)  # Clip to avoid numerical issues
    
    # Expected Improvement (for minimization)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (for minimization)
    pi = norm.cdf(z)
    
    # Thompson Sampling-like term: sample from GP posterior
    # Instead of a full sample, use a deterministic proxy:
    # The expected minimum of (mu + sigma * z) over z ~ N(0,1) is mu - sigma * E[max(0, z)]
    # But a simpler and effective proxy is to use the predictive lower confidence bound (LCB)
    # which is mu - beta * sigma. We use this as an exploration bonus.
    # To make it "Thompson-like", we use a fixed beta that balances exploration.
    beta_ts = 1.0
    ts_score = mu - beta_ts * sigma_safe  # Lower is better, so we'll negate later
    
    # Normalize EI, PI, and TS score to [0, 1] range for blending
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    pi_max = np.max(pi) if np.max(pi) > 1e-10 else 1.0
    ts_min = np.min(ts_score)
    ts_max = np.max(ts_score)
    ts_range = ts_max - ts_min if ts_range > 1e-10 else 1.0
    
    ei_norm = ei / ei_max
    pi_norm = pi / pi_max
    # For TS, lower ts_score is better, so we invert: (ts_max - ts_score) / ts_range
    ts_norm = (ts_max - ts_score) / ts_range
    
    # Blend EI, PI, and TS with weights
    # EI for exploitation, PI for moderate exploration, TS for aggressive exploration
    w_ei = 0.5
    w_pi = 0.3
    w_ts = 0.2
    
    blended = w_ei * ei_norm + w_pi * pi_norm + w_ts * ts_norm
    
    # Cost-aware: divide by cost (higher cost = lower score)
    scores = blended / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
