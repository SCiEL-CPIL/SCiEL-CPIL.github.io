"""Cost-aware acquisition for 2D three-hump camel BO. Fresh design."""
# EVOLVE-BLOCK-START
import time

import numpy as np
from scipy.stats import norm

# Global deadline guard: the evaluator kills the process at 330 s wall-clock.
# We want to finish comfortably before that, so reserve a safety margin.
_DEADLINE = time.time() + 300.0
_MARGIN = 5.0


def _time_ok():
    return time.time() < _DEADLINE - _MARGIN


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs so the output is always finite.
    sigma = np.clip(sigma, 1e-8, None)
    cost = np.clip(cost, 1e-8, None)
    f_best = float(f_best)

    # Expected improvement (in objective units).
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware discounting: divide by a smooth, increasing function of cost so
    # cheaper candidates are preferred. cost is in [1, 2] here (1 + mean of
    # coords scaled to [0,1]), so (1 + cost) gives a gentle, data-driven penalty
    # without needing a tuned hyperparameter.
    base = ei / (1.0 + cost)

    # UCB-style exploration term: reward places the GP is uncertain, again
    # cost-discounted so we don't pay a lot to explore.
    ucb = 0.3 * sigma / (1.0 + cost)

    # Diminishing exploration bonus: early on (large f_best) we explore more;
    # late on (small f_best) we focus on exploitation. f_best is in [0, ~50].
    explore_amt = 0.15 * np.sqrt(np.clip(f_best, 0.0, None))
    explore = explore_amt * (1.0 - cost / 2.0)  # cheaper = more exploration

    score = base + ucb + explore

    # Cheap deterministic refinement: among the top candidates, nudge toward the
    # one with the lowest predicted mean, breaking ties by lowest cost. This
    # gives a tiny, safe local-search effect without any heavy optimization.
    if _time_ok():
        top_n = 64
        idx = np.argpartition(-score, top_n)[:top_n]
        s_top = score[idx]
        m_top = mu[idx]
        c_top = cost[idx]
        # Rank by score, then prefer lower predicted mean and lower cost.
        order = np.lexsort((c_top, m_top, -s_top))
        best_idx = idx[order[0]]
        # Nudge: blend the refined candidate's score with the best one's,
        # but keep it below the global max so we don't create invalid spikes.
        s_max = np.max(score)
        score[best_idx] = np.minimum(s_max, s_max * 0.999 + 1e-9)

    # Final safety: guarantee finiteness.
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
