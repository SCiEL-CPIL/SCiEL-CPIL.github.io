"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _expected_improvement(mu, sigma, f_best):
    """Standard EI for minimization."""
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb(mu, sigma, f_best, xi=2.0):
    """Upper confidence bound for minimization (lower is better)."""
    return mu - xi * sigma


def _probability_of_improvement(mu, sigma, f_best):
    """Probability that x improves on f_best."""
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    return norm.cdf(z)


def _cost_adjusted_ei(mu, sigma, f_best, cost):
    """EI divided by cost, with a small floor on cost to avoid division issues."""
    ei = _expected_improvement(mu, sigma, f_best)
    c = np.maximum(cost, 1e-6)
    return ei / c


def _ucb_cost_aware(mu, sigma, f_best, cost, xi=2.0):
    """UCB scaled by 1/cost to prefer cheap exploratory points."""
    ucb = _ucb(mu, sigma, f_best, xi)
    c = np.maximum(cost, 1e-6)
    # Normalize UCB to be comparable: subtract f_best so positive = good
    return (f_best - ucb) / c


def acquisition(mu, sigma, f_best, cost):
    """
    Mixture acquisition: blend of cost-aware EI, cost-aware UCB, and POI.
    Weights shift from exploration (UCB, POI) to exploitation (EI) as
    f_best approaches the known minimum (0 for rosenbrock).
    """
    # Deadline guard: if we've used >280s, just return EI/cost quickly
    if _elapsed() > 280.0:
        return _cost_adjusted_ei(mu, sigma, f_best, cost)

    # Phase indicator: how close are we to the optimum (0 for rosenbrock)?
    # f_best starts high (~100s for rosenbrock) and decreases toward 0.
    # progress = 1 - f_best / f_initial. We don't know f_initial, so use
    # a heuristic: if f_best < 1, we're in late phase.
    f_best_val = float(f_best)
    # progress: 0 at start, 1 when f_best is very small
    progress = 1.0 - min(f_best_val / 50.0, 1.0)
    progress = np.clip(progress, 0.0, 1.0)

    # Weights: early -> more UCB/POI, late -> more EI
    w_ei = 0.3 + 0.5 * progress
    w_ucb = 0.5 - 0.3 * progress
    w_poi = 0.2 - 0.2 * progress

    # Normalize weights
    w_sum = w_ei + w_ucb + w_poi
    w_ei /= w_sum
    w_ucb /= w_sum
    w_poi /= w_sum

    # Compute each component
    ei_comp = _cost_adjusted_ei(mu, sigma, f_best, cost)
    ucb_comp = _ucb_cost_aware(mu, sigma, f_best, cost)
    poi_comp = _probability_of_improvement(mu, sigma, f_best)

    # Normalize each component to [0, 1] range for fair blending
    # EI and UCB are in "improvement per cost" units; POI is [0,1]
    # Scale EI and UCB to be comparable to POI by dividing by their max
    ei_max = np.max(ei_comp)
    if ei_max > 1e-9:
        ei_comp = ei_comp / ei_max
    ucb_max = np.max(ucb_comp)
    if ucb_max > 1e-9:
        ucb_comp = ucb_comp / ucb_max

    score = w_ei * ei_comp + w_ucb * ucb_comp + w_poi * poi_comp

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
