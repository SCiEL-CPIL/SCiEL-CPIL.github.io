"""Cost-aware acquisition: UCB-per-cost as the primary signal.
This approach uses a Upper Confidence Bound (UCB) term normalized by the
evaluation cost. This rewards high-uncertainty regions (high sigma) and
potential improvements (f_best - mu) while penalizing expensive evaluations.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# --- Explicit deadline guard (330 s hard kill; keep a safety margin) ---
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0   # 10 s safety margin before the 330 s kill

def _elapsed_ok():
    """True while we have at least ~3 s of margin before the hard limit."""
    return time.time() < _DEADLINE - 3.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization (minimization).

    mu, sigma, f_best, cost : each shape (2000,)
      f_best : scalar, lowest observed value
      cost   : per-candidate cost = 1 + mean(x_scaled)  (in [1, 2])
    Returns 2000 finite scores; higher = evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Robust floors -------------------------------------------------------
    cost_safe = np.maximum(cost, 1e-9)

    # Relative sigma floor to keep UCB meaningful when GP is very confident.
    # If sigma is extremely small, the exploration term vanishes.
    # We set a floor based on the spread of mu to ensure some exploration
    # remains even if the GP is overconfident.
    mu_min = float(np.min(mu))
    span = max(f_best - mu_min, 1e-9)
    sigma_floor_frac = 0.05
    sigma_floor = max(1e-6, sigma_floor_frac * span)
    sigma_safe = np.maximum(sigma, sigma_floor)

    # Deadline guard: fall back to a minimal valid score if near the limit.
    # Returning zeros is safe as it implies "no preference", but the evaluator
    # picks the max. If all are 0, it might pick the first one or fail.
    # Better to return a cheap, fast heuristic if we are close to the limit.
    if not _elapsed_ok():
        # Fallback: simple cost-normalized improvement
        score = (f_best - mu) / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # UCB for minimization: lower mu and higher sigma are better.
    # We want to maximize the potential improvement.
    # Improvement potential ~ f_best - mu
    # Uncertainty bonus ~ kappa * sigma
    # Total optimistic value ~ f_best - mu + kappa * sigma
    # We divide by cost to account for budget efficiency.
    
    # kappa = 2.0 was effective in the highest scoring inspiration.
    kappa = 2.0
    alpha = 1.0

    ucb = (f_best - mu) + kappa * sigma_safe
    cost_pen = cost_safe ** alpha
    score = ucb / cost_pen

    # Numerical safety: finite, non-negative.
    # Scores should be non-negative to avoid issues if the evaluator expects
    # positive weights, though "higher is better" is the only strict requirement.
    # Clipping to 0 ensures that if the GP predicts a value much worse than
    # f_best with low uncertainty, it doesn't get a negative score that might
    # be problematic in some selection mechanisms (though max selection handles
    # negatives fine, it's safer to keep them non-negative).
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
# EVOLVE-BLOCK-END
