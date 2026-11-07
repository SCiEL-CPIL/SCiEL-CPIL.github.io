import numpy as np
from scipy.stats import norm
import time

_acq_state = {'start_time': None, 'call_count': 0, 'total_cost': 0.0}

def acquisition(mu, sigma, f_best, cost):
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
        _acq_state['call_count'] = 0
        _acq_state['total_cost'] = 0.0
    _acq_state['call_count'] += 1
    
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 325.0:
        return np.zeros(len(mu), dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.5)
    
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Track total cost spent (approximate: each call evaluates 1 point)
    _acq_state['total_cost'] += np.mean(cost)
    
    # Remaining budget fraction: starts at 1.0, decreases as cost accumulates
    # Total budget is 30, initial cost ~5 (5 random points)
    total_budget = 30.0
    remaining_budget = max(total_budget - _acq_state['total_cost'], 1.0)
    budget_fraction = remaining_budget / total_budget  # 1.0 -> ~0.0
    
    # Call-based phase (for smooth transition)
    call_idx = min(_acq_state['call_count'], 50)
    call_phase = call_idx / 50.0
    
    # Combined phase: blend call-based and budget-based
    phase = 0.5 * call_phase + 0.5 * (1.0 - budget_fraction)
    
    # Cost-Adjusted LCB:
    # beta scales with remaining budget - more exploration when budget is high
    # beta = base * (0.5 + 0.5 * budget_fraction)
    # Early (high budget): beta ~ 2.0 * 1.0 = 2.0 (more exploration)
    # Late (low budget): beta ~ 2.0 * 0.5 = 1.0 (more exploitation)
    beta = 2.0 * (0.3 + 0.7 * budget_fraction)
    
    # LCB score for minimization: we want high score for low predicted values
    # lcb = mu - beta * sigma (lower is better for minimization)
    # Score = -lcb = -mu + beta * sigma
    lcb_score = -mu + beta * sigma
    
    # Cost penalty: exponential decay based on cost
    # cost ranges from ~1.1 to ~1.9
    # Normalize cost to [0, 1] range for penalty
    cost_norm = (cost - 1.0) / 1.0  # ~[0.1, 0.9]
    # Sharper penalty in late phase
    penalty_strength = 1.0 + 1.0 * phase  # 1.0 -> 2.0
    cost_penalty = np.exp(-penalty_strength * cost_norm)
    
    # Primary score: cost-adjusted LCB
    scores = lcb_score * cost_penalty
    
    # Add EI component for exploitation of known improvements
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # EI weight increases as budget depletes (more exploitation late)
    w_ei = 0.2 + 0.6 * phase  # 0.2 -> 0.8
    w_lcb = 1.0 - w_ei        # 0.8 -> 0.2
    
    # Scale EI by cost penalty too
    score_ei = ei * cost_penalty
    
    scores = w_lcb * (lcb_score * cost_penalty) + w_ei * score_ei
    
    # Small exploration bonus for high uncertainty in cheap regions
    exploration_bonus = 0.05 * sigma * cost_penalty * (1.0 - phase)
    scores = scores + exploration_bonus
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
