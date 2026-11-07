# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None
_start_time = None
_total_cost_budget = 30.0

def acquisition(mu, sigma, f_best, cost):
    """
    Multi-step cost-aware acquisition using a sequential decision framework.
    Estimates expected total regret reduction over remaining budget,
    combining immediate EI with information gain weighted by future potential.
    Higher score indicates a more promising candidate to evaluate next.
    """
    global _deadline, _start_time
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 320.0  # 10s safety margin

    # Check deadline
    if time.time() > _deadline:
        eps = 1e-10
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Estimate remaining budget from current cost vector
    # cost is the per-evaluation cost for each candidate
    mean_cost = np.mean(cost_safe)
    # Approximate number of evaluations remaining
    # We don't know exact total spent, but we can estimate from the structure
    # Use a heuristic: assume we're in the middle of the budget
    n_remaining_est = max(1.0, (_total_cost_budget * 0.5) / mean_cost)
    
    # Compute Z-scores for EI
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute information gain proxy: entropy reduction
    # High sigma regions have more to learn
    # Use normalized sigma as a proxy for information potential
    sigma_max = np.max(sigma_safe)
    if sigma_max > eps:
        info_gain = sigma_safe / sigma_max
    else:
        info_gain = np.zeros_like(sigma_safe)
    
    # Multi-step score:
    # Term 1: Immediate EI per cost (greedy)
    ei_per_cost = ei / cost_safe
    
    # Term 2: Future potential - evaluating cheap points in high-uncertainty regions
    # allows more evaluations, each of which can reduce regret
    # Weight by: (remaining evaluations possible) * (information gain) / cost
    future_potential = (n_remaining_est * info_gain) / cost_safe
    
    # Combine with adaptive weighting
    # When EI is large, trust the greedy term more
    # When EI is small, trust the exploration/future term more
    ei_scale = np.max(ei_per_cost) if np.max(ei_per_cost) > eps else 1.0
    ei_normalized = ei_per_cost / ei_scale
    
    # Blend: 70% immediate, 30% future potential
    scores = 0.7 * ei_per_cost + 0.3 * future_potential * ei_scale
    
    # Handle numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

# EVOLVE-BLOCK-END
