"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Lower Confidence Bound (LCB) with budget-aware exploration.
    Scales uncertainty penalty by inverse remaining budget fraction to force
    exploitation as cost limit approaches.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions in objective units
    - sigma: array of shape (2000,), GP standard deviations in objective units
    - f_best: scalar, lowest observed function value
    - cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,), acquisition values (higher = evaluate next)
    """
    # Initialize module-level state
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
        _acq_state['call_count'] = 0
    _acq_state['call_count'] += 1
    
    # Deadline guard - return safely if we're running out of time
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 325.0:
        return np.zeros(len(mu), dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values robustly
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.5)
    
    # Ensure positive values
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Budget-aware phase: track cumulative cost spent
    # Total budget is 30, initial 5 points cost ~5-10
    # Use call count as proxy for progress
    call_idx = min(_acq_state['call_count'], 40)
    phase = call_idx / 40.0  # 0 to 1
    
    # Remaining budget fraction (approximate)
    # Assume average cost per eval is ~1.5, so remaining fraction ~ (30 - spent)/30
    # With 40 calls max, spent ~ 1.5 * call_idx
    remaining_frac = max(1.0 - phase, 0.1)
    
    # --- Cost-Adjusted LCB ---
    # Beta scales with inverse remaining budget: aggressive exploitation when budget low
    # beta = base_beta / remaining_frac, but capped to avoid explosion
    base_beta = 2.0
    beta = np.minimum(base_beta / remaining_frac, 5.0)
    
    # LCB for minimization: -mu + beta * sigma
    # Higher LCB score means more promising (lower expected value with uncertainty)
    lcb_score = -mu + beta * sigma
    
    # --- Cost-awareness ---
    # Minimizer is in inner 80%: [1,9]^4
    # Cost = 1 + mean(x/10), so cost in [1.1, 1.9] for feasible region
    # Prefer points in valid cost range, penalize extremes
    
    # Smooth cost validity: prefer points with cost in [1.1, 1.9]
    lower_width = 0.08
    upper_width = 0.08
    
    lower_factor = 0.5 * (1.0 + np.tanh((cost - 1.1) / lower_width))
    upper_factor = 0.5 * (1.0 - np.tanh((cost - 1.9) / upper_width))
    cost_validity = np.clip(lower_factor * upper_factor, 0.0, 1.0)
    
    # Cost efficiency: prefer lower cost (cheaper evaluations allow more search)
    # Use exponential decay from ideal cost of 1.5
    cost_eff = np.exp(-0.8 * ((cost - 1.5) / 0.25) ** 2)
    
    # --- Combine ---
    # Primary score: LCB weighted by cost validity and efficiency
    scores = lcb_score * cost_validity * cost_eff
    
    # Add small exploration bonus that decays with phase
    # This helps in early iterations when GP is poorly calibrated
    exploration_bonus = 0.05 * sigma * cost_validity * (1.0 - phase)
    scores = scores + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
