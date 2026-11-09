import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = 320.0
_EPS = 1e-12


def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, _EPS)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb_score(mu, sigma, f_best, beta):
    """Compute UCB score for minimization: -mu + beta * sigma."""
    return -mu + beta * sigma


def _get_progress(eval_count, total_steps=25):
    """Get progress in [0, 1] over the 25 BO evaluation steps."""
    if eval_count <= 0:
        return 0.0
    return min(eval_count / float(total_steps), 1.0)


# Track evaluation count
_eval_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lookahead-UCB with smooth softmax gating.
    
    Uses a smooth temperature-based gating mechanism instead of a hard top-k
    threshold. The temperature adapts over time: high temperature early
    (broad exploration), low temperature late (sharp exploitation).
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - lowest observed value
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    global _eval_count
    
    # Deadline guard: fall back to pure EI
    if time.time() - _T0 > _DEADLINE:
        sigma_safe = np.maximum(sigma, _EPS)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)
    
    # Increment evaluation counter
    _eval_count += 1
    
    sigma_safe = np.maximum(sigma, _EPS)
    progress = _get_progress(_eval_count)
    
    # Adaptive UCB beta: start high (exploration), decay to low (exploitation)
    # beta from ~3.0 early to ~0.5 late
    beta = 3.0 - 2.5 * progress
    
    # Stage 1: UCB scores for coarse ranking
    ucb = _ucb_score(mu, sigma_safe, f_best, beta)
    
    # Stage 2: EI scores for refined selection
    ei = _expected_improvement(mu, sigma_safe, f_best)
    
    # Smooth gating: use temperature-controlled softmax on UCB scores
    # to create a smooth weight that gates the EI contribution
    # Temperature: high early (broad), low late (sharp)
    temperature = 5.0 - 4.5 * progress  # from 5.0 to 0.5
    
    # Normalize UCB scores for temperature application
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    
    if ucb_range < _EPS:
        # All UCB scores are essentially equal; uniform gating
        gate = np.ones_like(ucb)
    else:
        # Shift UCB to [0, 1] range
        ucb_norm = (ucb - ucb_min) / ucb_range
        # Apply temperature: lower temperature = sharper peak
        # Use scaled UCB with temperature in exponent
        ucb_scaled = ucb_norm * temperature
        # Softmax-like gating (numerically stable)
        ucb_scaled_max = np.max(ucb_scaled)
        exp_scaled = np.exp(ucb_scaled - ucb_scaled_max)
        gate = exp_scaled / np.sum(exp_scaled) * len(mu)
        # Normalize gate to [0, 1]
        gate_max = np.max(gate)
        if gate_max > _EPS:
            gate = gate / gate_max
    
    # Final score: blend of EI (exploitation) and UCB (exploration)
    # with smooth gating
    # EI dominates when gate is high (promising region), UCB provides
    # exploration pressure when gate is low
    score = gate * ei + (1.0 - gate) * ucb
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
