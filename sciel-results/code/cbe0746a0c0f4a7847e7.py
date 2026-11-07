"""Optimal market making with inventory risk using pre-computed HJB backward induction."""
import numpy as np
import time

# Module-level cache for policy tables
_policy_cache = {}
_start_time = time.time()
_TIME_LIMIT = 320  # safety margin before 330s kill


def _compute_policy_table(sigma, A, k, gamma, T, steps, Q, max_depth=5.0):
    """
    Compute optimal bid/ask depths via backward induction on the discrete-time model.
    Uses a two-phase approach: coarse grid for most steps, fine grid for early steps.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1)
        ask_table: array of shape (steps, 2*Q+1)
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_range = np.arange(-Q, Q + 1)
    
    # Value function
    V = np.zeros(n_inv)
    
    # Policy tables
    bid_table = np.full((steps, n_inv), np.inf)
    ask_table = np.full((steps, n_inv), np.inf)
    
    # Two-phase grid: coarse for most steps, fine for the last ~200 steps (early in time)
    n_coarse = 30
    n_fine = 100
    max_coarse = 3.0
    max_fine = 5.0
    
    # Pre-compute coarse grid
    depth_coarse = np.linspace(1e-6, max_coarse, n_coarse)
    exp_term_c = np.exp(-k * depth_coarse)
    rate_c = A * exp_term_c * dt
    p_fill_c = 1.0 - np.exp(-rate_c)
    
    # Pre-compute fine grid
    depth_fine = np.linspace(1e-6, max_fine, n_fine)
    exp_term_f = np.exp(-k * depth_fine)
    rate_f = A * exp_term_f * dt
    p_fill_f = 1.0 - np.exp(-rate_f)
    
    # Pre-compute 2D grids for coarse phase
    D_b_c, D_a_c = np.meshgrid(depth_coarse, depth_coarse, indexing='ij')
    P_b_c = 1.0 - np.exp(-A * np.exp(-k * D_b_c) * dt)
    P_a_c = 1.0 - np.exp(-A * np.exp(-k * D_a_c) * dt)
    earnings_c = P_b_c * D_b_c + P_a_c * D_a_c
    
    # Pre-compute 2D grids for fine phase
    D_b_f, D_a_f = np.meshgrid(depth_fine, depth_fine, indexing='ij')
    P_b_f = 1.0 - np.exp(-A * np.exp(-k * D_b_f) * dt)
    P_a_f = 1.0 - np.exp(-A * np.exp(-k * D_a_f) * dt)
    earnings_f = P_b_f * D_b_f + P_a_f * D_a_f
    
    # Threshold step: use fine grid for steps >= this value (closer to t=0)
    fine_threshold = steps - 200
    
    # Work backwards
    for n in range(steps - 1, -1, -1):
        # Check time limit
        if time.time() - _start_time > _TIME_LIMIT:
            break
        
        use_fine = (n >= fine_threshold)
        
        if use_fine:
            depth_grid = depth_fine
            p_fill = p_fill_f
            D_b = D_b_f
            D_a = D_a_f
            P_b = P_b_f
            P_a = P_a_f
            earnings = earnings_f
        else:
            depth_grid = depth_coarse
            p_fill = p_fill_c
            D_b = D_b_c
            D_a = D_a_c
            P_b = P_b_c
            P_a = P_a_c
            earnings = earnings_c
        
        for q_idx, q in enumerate(inv_range):
            # Inventory cost for this step
            inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            if q == Q:
                # Bid disabled, ask enabled
                q_next_ask_idx = (q - 1) + Q
                values = -depth_grid * p_fill + (1 - p_fill) * V[q_idx] + p_fill * V[q_next_ask_idx]
                best_idx = np.argmax(values)
                ask_table[n, q_idx] = depth_grid[best_idx]
                V[q_idx] = -inv_cost + values[best_idx]
                bid_table[n, q_idx] = np.inf
                
            elif q == -Q:
                # Ask disabled, bid enabled
                q_next_bid_idx = (q + 1) + Q
                values = -depth_grid * p_fill + (1 - p_fill) * V[q_idx] + p_fill * V[q_next_bid_idx]
                best_idx = np.argmax(values)
                bid_table[n, q_idx] = depth_grid[best_idx]
                V[q_idx] = -inv_cost + values[best_idx]
                ask_table[n, q_idx] = np.inf
                
            else:
                # Both sides available
                q_next_bid_idx = (q + 1) + Q
                q_next_ask_idx = (q - 1) + Q
                
                # Inventory value component
                inv_val = (P_b * P_a * V[q_idx] + 
                          P_b * (1 - P_a) * V[q_next_bid_idx] + 
                          (1 - P_b) * P_a * V[q_next_ask_idx] + 
                          (1 - P_b) * (1 - P_a) * V[q_idx])
                
                total = earnings + inv_val
                best_idx = np.unravel_index(np.argmax(total), total.shape)
                
                bid_table[n, q_idx] = depth_grid[best_idx[0]]
                ask_table[n, q_idx] = depth_grid[best_idx[1]]
                V[q_idx] = -inv_cost + total[best_idx]
    
    return bid_table, ask_table


def _get_policy_table(params):
    """Get or compute the policy table for given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        sigma, A, k, gamma = params['sigma'], params['A'], params['k'], params['gamma']
        T, steps, Q = params['T'], params['steps'], params['Q']
        
        bid_table, ask_table = _compute_policy_table(sigma, A, k, gamma, T, steps, Q)
        _policy_cache[key] = (bid_table, ask_table)
    
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    
    Args:
        t: current time (scalar)
        inventory: array of shape (4000,) with inventory levels
        mid: array of shape (4000,) with mid prices
        params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        (bid_depths, ask_depths): tuple of two arrays of shape (4000,)
    """
    # Determine current step
    dt = params['T'] / params['steps']
    step = int(round(t / dt))
    step = min(max(step, 0), params['steps'] - 1)
    
    Q = params['Q']
    
    # Get policy table
    bid_table, ask_table = _get_policy_table(params)
    
    # Map inventory to index
    inv_indices = inventory + Q
    inv_indices = np.clip(inv_indices, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[step, inv_indices].astype(float)
    ask_depths = ask_table[step, inv_indices].astype(float)
    
    # Ensure valid values: must be > -3/k, finite, non-negative
    min_valid = -3.0 / params['k']
    bid_depths = np.where((bid_depths > min_valid) & np.isfinite(bid_depths) & (bid_depths >= 0), bid_depths, np.inf)
    ask_depths = np.where((ask_depths > min_valid) & np.isfinite(ask_depths) & (ask_depths >= 0), ask_depths, np.inf)
    
    return bid_depths, ask_depths
