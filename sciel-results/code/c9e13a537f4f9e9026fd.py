"""Cost-aware acquisition for Bayesian optimization on 2D Rastrigin.

Combines:
- Cost-aware EI base score (EI / cost^alpha)
- UCB exploration term
- Pareto dominance factor (2-objective: EI vs cost)
- Rank-based final scoring to ensure smooth, finite output
"""
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE_SECONDS = 310.0  # Safety margin before 330s wall-clock limit


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)

    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.5, posinf=2.0, neginf=1.0)

    # Cost is in [1, 2] (1 + mean of x scaled to [0,1])
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)

    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Upper Confidence Bound for minimization (higher = more exploratory)
    kappa = 2.0
    ucb = -mu + kappa * sigma

    # Cost-aware base score: combine EI and UCB, penalize by cost
    # alpha=1.5 gives moderate cost sensitivity
    alpha = 1.5
    cost_safe = np.maximum(cost, 1e-12)

    # Weighted combination of exploitation (EI) and exploration (UCB)
    w_ei = 0.6
    w_ucb = 0.4
    raw_score = w_ei * ei + w_ucb * np.maximum(ucb, 0.0)

    # Apply cost penalty
    base_score = raw_score / (cost_safe ** alpha)

    # Pareto dominance factor on (ei, cost):
    # A point i dominates j if ei_i >= ei_j AND cost_i <= cost_j (strict in at least one)
    # For 2000 points, O(n^2) vectorized is ~4M comparisons, fast in NumPy
    n = len(mu)

    # Build dominance matrix
    ei_i = ei[:, None]       # (n, 1)
    ei_j = ei[None, :]       # (1, n)
    cost_i = cost[:, None]   # (n, 1)
    cost_j = cost[None, :]   # (1, n)

    # Non-inferiority: i is at least as good as j in both criteria
    non_inferior = (ei_i >= ei_j) & (cost_i <= cost_j)

    # Strict improvement in at least one dimension
    strict = (ei_i > ei_j) | (cost_i < cost_j)

    # Dominates: i dominates j
    dominates = non_inferior & strict

    # Count how many points dominate each point j (sum over axis=0)
    dom_count = np.sum(dominates, axis=0).astype(np.float64)

    # Pareto factor: 1.0 for non-dominated, decays smoothly for dominated
    # Using 1/sqrt(1+count) gives gentler decay than 1/(1+count)
    pareto_factor = 1.0 / np.sqrt(1.0 + dom_count)

    # Final score: base_score * pareto_factor
    score = base_score * pareto_factor

    # Rank-based transformation to ensure smooth ordering and finite output
    # Convert to rank, then use a power-law to emphasize top candidates
    ranks = np.argsort(np.argsort(score))  # 0 = lowest, n-1 = highest
    rank_norm = ranks / max(n - 1, 1)

    # Power law: p=1.5 emphasizes top candidates moderately
    p = 1.5
    final_score = rank_norm ** p

    # Scale to have meaningful magnitude
    final_score = final_score * 10.0

    # Ensure finite output
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1e6, neginf=-1e6)

    return final_score
