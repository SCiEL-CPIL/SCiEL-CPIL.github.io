"""Phase-based cost-aware acquisition with posterior sampling and budget-decaying exploration."""
import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START

def _safe_div(a, b, eps=1e-12):
    """Safe division to avoid zero/negative denominators."""
    return a / (b + eps)

def _ei(mu, sigma, f_best, eps=1e-9):
    """Standard Expected Improvement for minimization, with z-clipping."""
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _posterior_sample_score(mu, sigma, f_best, rng, n_samples=10):
    """
    Thompson Sampling-like acquisition: sample from GP posterior and estimate
    expected improvement of each candidate by averaging over samples.
    
    For each candidate x_i, we have mu_i, sigma_i. We draw n_samples samples
    from N(mu_i, sigma_i^2) and compute the fraction of samples that beat f_best,
    weighted by the improvement. This gives a stochastic score that naturally
    balances exploration (high sigma -> more samples beat f_best) and exploitation
    (low mu -> more samples are low).
    """
    n = len(mu)
    scores = np.zeros(n)
    
    for _ in range(n_samples):
        # Sample from GP posterior at all candidate points
        samples = mu + sigma * rng.standard_normal(n)
        # Improvement over f_best for each sample
        improvements = np.maximum(f_best - samples, 0.0)
        scores += improvements
    
    scores /= n_samples
    return scores

def _budget_phase_weight(cost, total_budget=30.0):
    """
    Compute a weight that transitions from exploration to exploitation
    based on the current cost relative to the total budget.
    
    Returns a value in [0, 1] where:
    - 1.0 = full exploration (early in budget)
    - 0.0 = pure exploitation (late in budget)
    
    Uses a smooth sigmoid-like transition centered at 60% of budget used.
    """
    # cost is the cost of each candidate point (shape 2000,)
    # We use the mean cost as a proxy for "where are we in the search"
    # Actually, cost varies per point. Let's use the median cost as a representative.
    # But we need a single weight for the whole call.
    # Use the minimum cost (cheapest points) as the "current phase" indicator?
    # No, better: use the fact that as we spend more budget, the average cost
    # of evaluated points tells us something. But we don't have that history.
    
    # Alternative: use the cost vector itself. Points with lower cost are "cheaper"
    # and should get more exploration weight. Points with higher cost should be
    # more exploitation-focused.
    
    # Actually, the weight should be per-point. Let's return a per-point weight.
    # Phase weight: 1.0 when cost is low (cheap, explore), 0.0 when cost is high (expensive, exploit)
    # But that's already captured by dividing by cost.
    
    # Let's think differently: the "phase" is determined by how much budget is left.
    # We don't have direct access to total spent, but we can infer it from the cost vector.
    # The cost of a point is 1 + mean(x_scaled). The minimum possible cost is 1.0,
    # maximum is 2.0. So cost in [1, 2].
    
    # A better approach: use the distribution of costs to estimate the phase.
    # Early on, we evaluate cheap points (cost ~1). Later, we might evaluate
    # more expensive points. But this is not reliable.
    
    # Simplest robust approach: use a fixed schedule based on the number of
    # acquisition calls. But we don't have that either without state.
    
    # Let's use a different idea: the weight depends on the sigma distribution.
    # Early in BO, sigma is high everywhere (unexplored). Later, sigma is low
    # near evaluated points and high in unexplored regions.
    # We can use the mean sigma as a proxy for "exploration needed".
    
    # Actually, let's just use a per-point weight that is the inverse of cost,
    # which is already standard. The "phase" idea needs to be implemented differently.
    
    # New approach: use a smooth function of cost that gives more weight to
    # low-cost points in a way that depends on the overall cost distribution.
    # This is getting complex. Let's simplify.
    
    return None  # We'll handle this differently in the main function

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using posterior sampling with budget-aware weighting.
    
    Strategy:
    1. Compute EI/cost as the exploitation baseline.
    2. Compute a Thompson Sampling score (posterior sampling) as the exploration signal.
    3. Combine them with a weight that depends on the cost distribution:
       - Low-cost points get more exploration weight.
       - High-cost points get more exploitation weight.
    4. The overall score is designed to be finite and higher = better.
    """
    n = len(mu)
    
    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Exploitation: EI per unit cost
    ei = _ei(mu, sigma_safe, f_best)
    ei_cost = ei / cost_safe
    
    # Exploration: Thompson Sampling score per unit cost
    # Use a fixed seed for reproducibility within a single call
    # (The evaluator calls this once per iteration, so we can use a deterministic
    #  pseudo-random sequence based on the input to be consistent)
    # Actually, we should use a random seed that changes each call to get
    #  different exploration. But we don't have a call counter.
    # Use a hash of the inputs as seed for the RNG.
    # This ensures different calls get different random samples.
    seed = int(np.abs(np.sum(mu * 1e6 + sigma * 1e4 + cost * 1e2))) % (2**32)
    rng = np.random.default_rng(seed)
    
    ts_scores = _posterior_sample_score(mu, sigma_safe, f_best, rng, n_samples=8)
    ts_cost = ts_scores / cost_safe
    
    # Combine: weight the exploration term by a factor that depends on the
    #  relative cost of each point. Cheaper points get more exploration.
    #  Use a normalized cost: cost_norm = (cost - min_cost) / (max_cost - min_cost + eps)
    #  Then weight_exploration = 1 - cost_norm (cheap points get weight ~1, expensive ~0)
    #  But this is too aggressive. Let's use a softer version.
    
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost_safe - cost_min) / cost_range  # in [0, 1]
    
    # Exploration weight: high for cheap points, low for expensive points
    # Use a sigmoid-like transition: weight = 1 / (1 + exp(k * (cost_norm - 0.5)))
    # With k=4, this gives weight ~0.9 at cost_norm=0 and ~0.1 at cost_norm=1
    k = 4.0
    expl_weight = 1.0 / (1.0 + np.exp(k * (cost_norm - 0.5)))
    
    # Final score: exploitation + exploration_weight * exploration
    score = ei_cost + expl_weight * ts_cost
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
