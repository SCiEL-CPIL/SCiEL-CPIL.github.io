"""Cost-aware acquisition for Bayesian optimization on Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We keep a safety margin so we return before the hard limit.
_DEADLINE = 320.0  # seconds from module load (approximate; per-call overhead is small)
_start_time = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite acquisition scores (higher = evaluate next).

    mu, sigma: shape (2000,), GP mean and std in objective units.
    f_best: scalar, lowest observed value.
    cost: shape (2000,), cost of evaluating each candidate.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Return a safe fallback: uniform random scores
        return np.random.uniform(0, 1, size=len(mu))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Adaptive UCB-like hybrid with smooth cost-aware normalization
    # As f_best approaches 0 (true minimum), reduce exploration weight
    # to focus on exploitation. Use tanh to bound scores and prevent
    # extreme values from dominating the selection.
    
    # Scale beta based on progress: when f_best is large, explore more;
    # when f_best is close to 0, exploit more.
    # f_best starts around 2-3 for Ackley and decreases to ~0.
    progress = np.clip(1.0 - f_best / 5.0, 0.0, 1.0)
    beta = 1.5 - 0.8 * progress  # beta ranges from 1.5 (early) to 0.7 (late)
    
    acq = (f_best - mu) + beta * sigma
    
    # Cost-aware: divide by cost to get improvement per unit cost
    acq = acq / cost
    
    # Smooth the acquisition to prevent extreme values from dominating.
    # Use tanh to bound scores to roughly [-1, 1], then rescale.
    # This helps when a few points have very high/low scores.
    acq_smooth = np.tanh(acq)
    
    # Ensure finite values
    acq_smooth = np.where(np.isfinite(acq_smooth), acq_smooth, 0.0)
    
    return acq_smooth
# EVOLVE-BLOCK-END
