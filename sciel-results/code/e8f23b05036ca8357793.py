import numpy as np
from scipy.stats import norm
import time

# Global state to track progress across calls within a seed
_step_count = 0
_start_time = None

def _quantile_acquisition(mu, sigma, f_best):
    """Quantile Acquisition Function: P(X < f_best) = Phi((f_best - mu) / sigma)"""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    return norm.cdf(z)

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Standard Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _ucb(mu, sigma, kappa):
    """Lower Confidence Bound for minimization (we maximize the negative)."""
    return -mu + kappa * sigma

def acquisition(mu, sigma, f_best):
    global _step_count, _start_time
    
    # Initialize time tracking on first call
    if _start_time is None:
        _start_time = time.time()
        _step_count = 0
    
    elapsed = time.time() - _start_time
    _step_count += 1
    
    # Safety margin: return UCB if we're getting close to deadline
    if elapsed > 300:
        scores = -mu + 2.0 * sigma
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Compute component scores
    qaf_scores = _quantile_acquisition(mu, sigma, f_best)
    ei_scores = _expected_improvement(mu, sigma, f_best, xi=0.01)
    ucb_scores = _ucb(mu, sigma, kappa=2.0)
    
    # Normalize each score to [0, 1] for combination
    def normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-12:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    qaf_norm = normalize(qaf_scores)
    ei_norm = normalize(ei_scores)
    ucb_norm = normalize(ucb_scores)
    
    # Adaptive weighting based on progress
    # Early on: more exploration (UCB, QAF)
    # Later: more exploitation (EI)
    progress = min(1.0, _step_count / 25.0)  # 25 steps to fully exploit
    
    # Weights: QAF, EI, UCB
    w_qaf = 0.4 * (1 - 0.5 * progress)
    w_ei = 0.3 + 0.4 * progress
    w_ucb = 0.3 * (1 - 0.5 * progress)
    
    # Combine
    scores = w_qaf * qaf_norm + w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return scores
