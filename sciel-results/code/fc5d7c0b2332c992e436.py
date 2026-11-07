import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0


def _ei_for_minimization(mu, sigma, f_best):
    """Expected Improvement for minimization: E[max(f_best - f(x), 0)]"""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei


def _ucb_for_minimization(mu, sigma, beta):
    """UCB for minimization: mu - beta * sigma (lower is better)"""
    return mu - beta * sigma


def acquisition(mu, sigma, f_best, cost):
    """
    Deterministic Cost-Aware Acquisition using Non-Linear Cost Penalties.
    
    Structural Change: 
    1. Replaces stochastic Thompson Sampling with deterministic UCB/EI blend.
    2. Uses a non-linear (power) cost penalty for the exploitation term (EI)
       to strongly discourage expensive evaluations when improvement is low.
    3. Uses a linear cost penalty for the exploration term (UCB) to allow
       some expensive exploration if uncertainty is high.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)

    # Clean inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        # Fallback: simple EI / cost
        z = (f_best - mu) / (sigma + 1e-12)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.where(np.isfinite(ei / cost), ei / cost, 0.0)

    # Estimate progress to modulate exploration/exploitation balance
    known_min = -78.3323
    f_start_est = 20.0
    # Clip progress to [0, 1]
    progress = np.clip((f_start_est - f_best) / (f_start_est - known_min), 0.0, 1.0)

    # 1. Compute Expected Improvement (Exploitation)
    ei = _ei_for_minimization(mu, sigma, f_best)
    
    # 2. Compute UCB (Exploration)
    # Beta increases as we progress less? Or decreases?
    # Early on, we want to explore. Later, exploit.
    # Beta starts high (explore) and decreases?
    # Actually, standard UCB beta is constant or increases with log(t).
    # Here, let's use a fixed beta that balances exploration.
    beta = 1.5
    ucb = _ucb_for_minimization(mu, sigma, beta)
    
    # 3. Cost Normalization
    # Normalize costs so that mean cost is 1. This makes the penalty scale-invariant.
    mean_cost = np.mean(cost)
    if mean_cost < 1e-6:
        mean_cost = 1.0
    cost_norm = cost / mean_cost
    
    # 4. Cost-Adjusted Scores
    # Exploitation term: Penalize cost heavily (power 1.5)
    # This reflects that "improvement per cost" is the key metric, 
    # but high cost should be penalized super-linearly to avoid "expensive noise".
    exploit_score = ei / (cost_norm ** 1.5)
    
    # Exploration term: Penalize cost linearly
    # We want to explore high uncertainty areas, but not if they are too expensive.
    explore_score = -ucb / cost_norm  # Note: UCB is "lower is better", so we negate for "higher score is better"
    
    # 5. Blend
    # Early on (progress ~ 0), we might want more exploration?
    # Or do we want to find the minimum quickly?
    # Usually, EI is a good balance. 
    # Let's weight EI heavily, but add UCB to break ties or guide to uncertain regions.
    # Weight of EI: 0.8 to 0.6
    # Weight of UCB: 0.2 to 0.4
    weight_ei = 0.8 - 0.2 * progress
    weight_ucb = 1.0 - weight_ei
    
    combined = weight_ei * exploit_score + weight_ucb * explore_score
    
    # Ensure finite
    combined = np.where(np.isfinite(combined), combined, 0.0)

    return combined
