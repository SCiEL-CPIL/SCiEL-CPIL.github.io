# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants
_N_CAND = 2000

# Global state to track total evaluations for adaptive exploration
_total_cost_accumulated = 0.0
_INITIAL_COST = 5.0  # 5 initial uniform random points

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a knowledge-gradient-inspired approach: combines probability of
    significant improvement (PSI) with expected improvement per unit cost,
    with an adaptive exploration coefficient that decays with total cost consumed.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - higher values are preferred
    """
    global _total_cost_accumulated
    
    # Deadline guard: 330s limit, start timer
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin

    # Ensure arrays are 1D and float
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    score_ei = ei / cost_safe
    
    # Probability of Significant Improvement (PSI) - knowledge gradient proxy
    # P(f(x) < f_best - delta) where delta scales with current regret
    delta = max(0.01 * abs(f_best), 1e-6)
    z_psi = (f_best - delta - mu) / sigma_safe
    psi = norm.cdf(z_psi)
    
    # Cost-aware PSI
    psi_per_cost = psi / cost_safe
    
    # Adaptive exploration coefficient that decays with total cost consumed
    # The total budget is 30, so as we consume more cost, we should explore less
    total_cost_consumed = _total_cost_accumulated
    remaining_budget_fraction = max(0.0, 1.0 - total_cost_consumed / 30.0)
    
    # Exploration coefficient: starts high when budget is fresh, decays as budget is consumed
    # Use a smooth decay: beta = base_beta * (remaining_budget_fraction + epsilon)
    base_beta = 2.0
    epsilon = 0.1
    beta = base_beta * (remaining_budget_fraction + epsilon)
    
    # Lower Confidence Bound (LCB) component
    # LCB = mu - beta * sigma (for minimization, we want low mu and high sigma)
    # Cost-aware LCB: (mu - beta * sigma) / cost
    lcb = mu - beta * sigma_safe
    lcb_per_cost = -lcb / cost_safe  # Negate because we want higher scores for lower LCB
    
    # Normalize components to comparable scale
    max_ei_per_cost = np.max(score_ei)
    if max_ei_per_cost > 1e-12:
        ei_per_cost_norm = score_ei / max_ei_per_cost
    else:
        ei_per_cost_norm = score_ei
    
    max_lcb_per_cost = np.max(lcb_per_cost)
    if max_lcb_per_cost > 1e-12:
        lcb_per_cost_norm = lcb_per_cost / max_lcb_per_cost
    else:
        lcb_per_cost_norm = lcb_per_cost
    
    max_psi_per_cost = np.max(psi_per_cost)
    if max_psi_per_cost > 1e-12:
        psi_per_cost_norm = psi_per_cost / max_psi_per_cost
    else:
        psi_per_cost_norm = psi_per_cost
    
    # Dynamic weights: 
    # - When f_best is high (early), weight LCB and PSI more for exploration
    # - When f_best is low (late), weight EI more for exploitation
    f_best_abs = abs(f_best)
    
    # Weight for EI (exploitation): increases as f_best decreases
    w_ei = 0.5 + 0.5 * (1.0 / (1.0 + f_best_abs / 2.0))
    
    # Weight for LCB (exploration via uncertainty): decreases as budget is consumed
    w_lcb = 0.3 * remaining_budget_fraction
    
    # Weight for PSI (exploration via probability of improvement): moderate
    w_psi = 0.2 * (1.0 - 0.5 * remaining_budget_fraction)
    
    # Normalize weights to sum to 1
    w_sum = w_ei + w_lcb + w_psi
    if w_sum < 1e-9:
        w_sum = 1e-9
    w_ei /= w_sum
    w_lcb /= w_sum
    w_psi /= w_sum
    
    # Combine components
    scores = w_ei * ei_per_cost_norm + w_lcb * lcb_per_cost_norm + w_psi * psi_per_cost_norm
    
    # Add a small exploration bonus for high uncertainty regions
    max_sigma = np.max(sigma_safe)
    if max_sigma > 1e-12:
        exploration_bonus = 0.05 * (sigma_safe / max_sigma)
    else:
        exploration_bonus = 0.0
    
    scores = scores + exploration_bonus
    
    # Check time
    if time.time() > deadline:
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
