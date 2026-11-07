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
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1)
        ask_table: array of shape (steps, 2*Q+1)
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_range = np.arange(-Q, Q + 1)
    
    # Value function: V[n, q] = optimal expected objective from step n with inventory q
    # We work backwards from n = steps-1 down to n = 0
    
    # Terminal value: no terminal charge
    V = np.zeros(n_inv)
    
    # We need to store the policy at each step
    bid_table = np.full((steps, n_inv), np.inf)
    ask_table = np.full((steps, n_inv), np.inf)
    
    # Pre-compute the 2D meshgrid and probabilities once (constant across all steps)
    n_grid_2d = 80
    depth_grid_2d = np.linspace(1e-6, max_depth, n_grid_2d)
    
    D_b, D_a = np.meshgrid(depth_grid_2d, depth_grid_2d, indexing='ij')
    P_b = 1.0 - np.exp(-A * np.exp(-k * D_b) * dt)
    P_a = 1.0 - np.exp(-A * np.exp(-k * D_a) * dt)
    
    # Pre-compute earnings (constant across all steps and inventory levels)
    earnings_2d = P_b * D_b + P_a * D_a
    
    # For 1D boundary cases (q=Q or q=-Q), use a finer 1D grid
    n_grid_1d = 300
    depth_grid_1d = np.linspace(1e-6, max_depth, n_grid_1d)
    p_fill_1d = 1.0 - np.exp(-A * np.exp(-k * depth_grid_1d) * dt)
    earnings_1d = depth_grid_1d * p_fill_1d
    
    # Work backwards
    for n in range(steps - 1, -1, -1):
        # Check time limit
        if time.time() - _start_time > _TIME_LIMIT:
            # Return what we have (remaining entries stay inf)
            break
        
        V_next = V.copy()
        
        for q_idx, q in enumerate(inv_range):
            # Inventory cost for this step
            inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            if q == Q:
                # Only ask side available
                # Ask fill: earn d, inventory becomes q-1 = Q-1
                q_next_ask_idx = (q - 1) + Q
                v_q = V_next[q_idx]
                v_qa = V_next[q_next_ask_idx]
                
                # Maximize over d: -d*p(d) + (1-p(d))*v_q + p(d)*v_qa
                # = (1-p(d))*v_q + p(d)*(v_qa - d)
                # = v_q + p(d)*(v_qa - d - v_q)
                values = earnings_1d + (1.0 - p_fill_1d) * v_q + p_fill_1d * (v_qa - earnings_1d / p_fill_1d)
                # Actually simpler: values = (1-p)*v_q + p*v_qa - d*p
                # = v_q + p*(v_qa - v_q) - d*p
                values = v_q + p_fill_1d * (v_qa - v_q) - earnings_1d
                best_idx = np.argmax(values)
                ask_table[n, q_idx] = depth_grid_1d[best_idx]
                V[q_idx] = -inv_cost + values[best_idx]
                
                bid_table[n, q_idx] = np.inf
                
            elif q == -Q:
                # Bid enabled, ask disabled
                q_next_bid_idx = (q + 1) + Q
                v_q = V_next[q_idx]
                v_qb = V_next[q_next_bid_idx]
                
                # Maximize over d: -d*p(d) + (1-p(d))*v_q + p(d)*v_qb
                # = v_q + p*(v_qb - v_q) - d*p
                values = v_q + p_fill_1d * (v_qb - v_q) - earnings_1d
                best_idx = np.argmax(values)
                bid_table[n, q_idx] = depth_grid_1d[best_idx]
                V[q_idx] = -inv_cost + values[best_idx]
                
                ask_table[n, q_idx] = np.inf
                
            else:
                # Both sides available
                q_next_bid_idx = (q + 1) + Q
                q_next_ask_idx = (q - 1) + Q
                
                v_q = V_next[q_idx]
                v_qb = V_next[q_next_bid_idx]
                v_qa = V_next[q_next_ask_idx]
                
                # Inventory value term:
                # P_b*P_a*v_q + P_b*(1-P_a)*v_qb + (1-P_b)*P_a*v_qa + (1-P_b)*(1-P_a)*v_q
                inv_val = (P_b * P_a * v_q + 
                          P_b * (1.0 - P_a) * v_qb + 
                          (1.0 - P_b) * P_a * v_qa + 
                          (1.0 - P_b) * (1.0 - P_a) * v_q)
                
                total = earnings_2d + inv_val
                best_idx = np.unravel_index(np.argmax(total), total.shape)
                
                bid_table[n, q_idx] = depth_grid_2d[best_idx[0]]
                ask_table[n, q_idx] = depth_grid_2d[best_idx[1]]
                V[q_idx] = -inv_cost + total[best_idx]
        
        V_next = V  # for next iteration
    
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
