"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar
import time

_cache = {}

def _get_dp_tables(params):
    """Compute optimal bid/ask depth tables via dynamic programming."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Value table: V[n, q] for n=0..steps, q=-Q..Q
    # V[steps][q] = 0 (no terminal charge)
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Optimal depth tables
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)
    
    # Precompute risk penalty
    q_indices = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma**2 * q_indices**2 * dt
    
    def fill_prob(depth):
        """Compute fill probability for a given depth."""
        if depth <= 0 or not np.isfinite(depth):
            return 0.0
        return 1.0 - np.exp(-A * np.exp(-k * depth) * dt)
    
    def max_bid_reward(delta_V, q_idx):
        """Maximize p(d) * (d + delta_V) over d >= 0."""
        if q_idx >= Q:  # at max inventory, no bid fill
            return 0.0, np.inf
        
        def neg_obj(d):
            if d <= 0:
                return 0.0
            p = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
            return -(p * (d + delta_V))
        
        # Golden section search on [0, 10/k]
        # The optimal depth shouldn't be too large
        d_max = 10.0 / k
        
        result = minimize_scalar(neg_obj, bounds=(0.0, d_max), method='bounded')
        d_opt = max(0.0, result.x)
        reward = -result.fun if np.isfinite(result.fun) else 0.0
        
        return reward, d_opt
    
    def max_ask_reward(delta_V, q_idx):
        """Maximize p(d) * (d + delta_V) over d >= 0."""
        if q_idx <= -Q:  # at min inventory, no ask fill
            return 0.0, np.inf
        
        def neg_obj(d):
            if d <= 0:
                return 0.0
            p = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
            return -(p * (d + delta_V))
        
        d_max = 10.0 / k
        
        result = minimize_scalar(neg_obj, bounds=(0.0, d_max), method='bounded')
        d_opt = max(0.0, result.x)
        reward = -result.fun if np.isfinite(result.fun) else 0.0
        
        return reward, d_opt
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        for q_idx in range(-Q, Q + 1):
            V_next_q = V[n + 1, q_idx + Q]
            
            # Bid side
            if q_idx < Q:
                delta_b = V[n + 1, q_idx + 1 + Q] - V_next_q
                bid_reward, bid_depth = max_bid_reward(delta_b, q_idx)
            else:
                bid_reward = 0.0
                bid_depth = np.inf
            
            # Ask side
            if q_idx > -Q:
                delta_a = V[n + 1, q_idx - 1 + Q] - V_next_q
                ask_reward, ask_depth = max_ask_reward(delta_a, q_idx)
            else:
                ask_reward = 0.0
                ask_depth = np.inf
            
            V[n, q_idx + Q] = V_next_q - risk[q_idx + Q] + bid_reward + ask_reward
            opt_bid[n, q_idx + Q] = bid_depth
            opt_ask[n, q_idx + Q] = ask_depth
    
    _cache[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """Quote optimal depths based on dynamic programming solution."""
    start_time = time.time()
    
    opt_bid, opt_ask = _get_dp_tables(params)
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    # Current step index
    n = int(t / dt)
    n = min(n, steps - 1)  # clamp to valid range
    
    # Map inventory to indices
    inv_idx = inventory + Q
    
    bid_depths = opt_bid[n, inv_idx]
    ask_depths = opt_ask[n, inv_idx]
    
    # Safety: ensure valid values
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 0), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 0), ask_depths, np.inf)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)

# EVOLVE-BLOCK-END
