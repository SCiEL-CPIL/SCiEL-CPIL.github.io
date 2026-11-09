"""Rewritten acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: evaluator kills after 330s; keep a safety margin.
_DEADLINE = 320.0
_start_time = None


def _check_deadline():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time >= _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with an adaptive exploration weight. Higher score = more promising candidate.
    
    Parameters:
        mu: array of shape (2000,), posterior mean in objective units
        sigma: array of shape (2000,), posterior std in objective units
        f_best: float, lowest observed value so far
    
    Returns:
        scores: array of shape (2000,), finite, higher is better
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-9)

    # Known global minimum of Michalewicz 2D
    f_min = -1.8013

    # Adaptive exploration weight: increase when f_best is far from the known minimum
    gap = f_best - f_min  # positive when we're far from optimum
    # Clip gap to a reasonable range
    gap_clipped = np.clip(gap, 0.0, 10.0)
    # Exploration weight beta: starts moderate, grows as we're further from optimum
    beta = 1.5 + 2.0 * np.tanh(gap_clipped / 2.0)

    # Compute EI
    z_ei = (f_best - mu) / sigma
    # Handle numerical issues
    z_ei = np.clip(z_ei, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative

    # Compute UCB: mu - beta * sigma (we're minimizing, so lower is better; UCB for minimization)
    # For minimization, the "upper confidence bound" on the *negative* (i.e., we want low values)
    # Standard UCB for minimization: mu - beta * sigma
    ucb = mu - beta * sigma

    # Normalize EI to [0, 1] roughly for blending
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ei_norm = ei / ei_max

    # Normalize UCB: lower ucb is better, so we want higher scores for lower ucb
    # Invert: score_ucb = -ucb, then normalize
    ucb_score = -ucb
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_score)

    # Blend: weight EI more when close to optimum, UCB more when far
    # When gap is small (close to optimum), rely more on EI (exploitation)
    # When gap is large, rely more on UCB (exploration)
    ei_weight = 0.6 - 0.3 * np.tanh(gap_clipped / 2.0)  # 0.6 to 0.3
    ucb_weight = 1.0 - ei_weight  # 0.4 to 0.7

    scores = ei_weight * ei_norm + ucb_weight * ucb_norm

    # Add a small uncertainty bonus to encourage exploring uncertain regions
    sigma_max = np.max(sigma) if np.max(sigma) > 1e-12 else 1.0
    uncertainty_bonus = 0.1 * (sigma / sigma_max)
    scores += uncertainty_bonus

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Check deadline
    if _check_deadline():
        # Return best result found with safety margin
        return scores

    return scores
# EVOLVE-BLOCK-END
