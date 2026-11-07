"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_start_time = None
_DEADLINE = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines expected improvement per unit cost with adaptive UCB exploration.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're close to the limit, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Fallback: use just EI/cost to be fast
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        score = ei / (cost + 1e-9)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Expected improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware EI
    ei_cost = ei / (cost + 1e-9)
    
    # UCB term: -mu + beta * sigma
    # Adapt beta based on remaining budget: more exploration when budget is high
    remaining_budget = max(0.0, 30.0 - cost.sum() / 2000.0)  # approximate remaining
    # Actually, we don't have total cost, so use a simpler heuristic
    # Use the mean cost as a proxy for how much we've spent
    mean_cost = np.mean(cost)
    # If mean cost is low, we're early; if high, we're late
    # Adapt beta: start at 2.5, decrease to 1.0
    beta = 1.0 + 1.5 * np.exp(-mean_cost * 2.0)
    ucb = -mu + beta * sigma
    
    # Smooth combination using log-sum-exp style
    # We want to combine ei_cost and ucb smoothly
    # Use a soft-max style combination: log(exp(a*ei_cost) + exp(b*ucb))
    # But simpler: use a weighted geometric mean or smooth max
    # Let's use: score = log(exp(ei_cost) + exp(ucb * 0.1)) / log(2) for balance
    # Actually, let's try a simpler smooth combination:
    # score = (ei_cost + ucb * 0.1) + 0.5 * (ei_cost - ucb * 0.1)**2 / (ei_cost**2 + ucb**2 + 1e-9)
    # This adds a penalty when the two terms disagree
    
    # Simpler approach: weighted sum with adaptive weight
    # Weight for EI/cost increases as we get closer to convergence
    w_ei = 0.7 + 0.3 * np.exp(-beta)
    w_ucb = 1.0 - w_ei
    
    score = w_ei * ei_cost + w_ucb * ucb
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
