"""
Optimized market making policy with tuned neutral depth and time-decaying skew.
Uses a pre-computed optimal neutral depth for each regime and applies an inventory-based skew that decays to zero at the horizon, aligning with the theoretical optimal control solution where inventory risk becomes irrelevant in the final step.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_T0 = time.time()
_DEADLINE = 290.0

# Cache for optimal neutral depth per regime
_OPT_DEPTH_CACHE = {}

def _remaining_budget():
    return _DEADLINE - (time.time() - _T0)

def _get_optimal_neutral_depth(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the depth that maximizes expected depth earned per step for a neutral player.
    f(d) = d * (1 - exp(-A * exp(-k*d) * dt))
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _OPT_DEPTH_CACHE:
        return _OPT_DEPTH_CACHE[key]
    
    if _remaining_budget() < 1.0:
        # Fallback to analytic approximation
        _OPT_DEPTH_CACHE[key] = 1.0 / k
        return _OPT_DEPTH_CACHE[key]
        
    dt = T / steps
    # Search for optimal depth in [0, 3/k]
    # The function is unimodal.
    # Use a fine grid or golden section. Grid is safer.
    d_min = 0.0
    d_max = 3.0 / k
    num_points = 100
    d_vals = np.linspace(d_min, d_max, num_points)
    
    # Calculate expected depth earned
    # P = 1 - exp(-A * exp(-k*d) * dt)
    # Earned = d * P
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * d_vals) * dt)
    earned = d_vals * fill_probs
    
    idx = np.argmax(earned)
    # Refine around the best point
    if idx > 0 and idx < num_points - 1:
        d_left = d_vals[idx-1]
        d_right = d_vals[idx+1]
        d_fine = np.linspace(d_left, d_right, 20)
        fill_probs_fine = 1.0 - np.exp(-A * np.exp(-k * d_fine) * dt)
        earned_fine = d_fine * fill_probs_fine
        idx_fine = np.argmax(earned_fine)
        opt_depth = d_fine[idx_fine]
    else:
        opt_depth = d_vals[idx]
        
    _OPT_DEPTH_CACHE[key] = opt_depth
    return opt_depth

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    if _remaining_budget() < 0.5:
        # Emergency fallback
        max_depth = 3.0 / k
        bid = np.full(len(inventory), 0.5 * max_depth)
        ask = np.full(len(inventory), 0.5 * max_depth)
        return bid, ask

    # Get optimal neutral depth
    base_depth = _get_optimal_neutral_depth(sigma, A, k, gamma, T, steps, Q)
    
    inv = inventory.astype(np.float64)
    
    # Time decay factor: 1 at t=0, 0 at t=T
    remaining_fraction = max(0.0, 1.0 - t / T)
    
    # Skew factor
    # Using the heuristic coefficient, scaled by time remaining
    skew_factor = (gamma * sigma**2) / (A * k) * remaining_fraction
    skew = skew_factor * inv
    
    # Compute depths
    bid_depths = base_depth * (1.0 + skew)
    ask_depths = base_depth * (1.0 - skew)
    
    # Clip to valid range
    min_depth = 1e-6
    max_depth = 3.0 / k
    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)
    
    # Handle inventory bounds
    bid_depths[inv >= Q] = np.inf
    ask_depths[inv <= -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
