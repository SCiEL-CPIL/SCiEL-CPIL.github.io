"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_ACQ_START_TIME = time.time()
_ACQ_DEADLINE = _ACQ_START_TIME + 280
_ACQ_CALL_COUNT = [0]


def acquisition(mu, np_sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rosenbrock 2d.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - np_sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: scalar, best observed value so far
    - cost: array of shape (2000,), evaluation costs for each candidate
    
    Returns:
    - array of shape (2000,), acquisition scores (higher = better)
    """
    _ACQ_CALL_COUNT[0] += 1

    # Deadline guard
    if time.time() > _ACQ_DEADLINE:
        fallback = -mu / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(fallback), fallback, 0.0)

    # Sanitize inputs
    sigma = np.maximum(np_sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)

    # Cost-Aware UCB: score = (f_best - mu + beta * sigma) / cost
    # We want to maximize the upper confidence bound of improvement over cost.
    # Since we are minimizing, the "upper bound" of the function value is mu + beta*sigma.
    # We want low values, so we score by how much lower the UCB is than f_best, divided by cost.
    # Actually, standard UCB for minimization: minimize mu + beta*sigma.
    # Acquisition score should be high for good candidates.
    # Good candidate: low mu + beta*sigma, low cost.
    # Let's use: score = -(mu + beta * sigma) / cost
    # Or equivalently, maximize (f_best - (mu + beta * sigma)) / cost, which rewards expected improvement.
    # But the inspiration says "linear Cost-Aware UCB".
    # Let's stick to a simple linear form: (f_best - mu + beta * sigma) / cost is not quite UCB.
    # UCB for minimization is mu + beta * sigma. We want to minimize this.
    # So we want to maximize - (mu + beta * sigma) / cost.
    
    # Adaptive beta: increase exploration early, decrease later.
    n_calls = _ACQ_CALL_COUNT[0]
    # Beta decays from 1.0 to 0.1
    beta = 1.0 * np.exp(-n_calls / 50.0) + 0.1

    # Compute UCB score
    # score = - (mu + beta * sigma) / cost
    # This naturally balances exploitation (low mu) and exploration (high sigma) against cost.
    ucb_score = - (mu + beta * sigma) / cost

    # Ensure finiteness
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    return ucb_score
# EVOLVE-BLOCK-END
