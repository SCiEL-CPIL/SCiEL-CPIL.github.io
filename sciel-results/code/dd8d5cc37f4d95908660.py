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
    
    # Grid of candidate depths to search over
    # Use a fine grid for accuracy
    n_grid = 200
    depth_grid = np.linspace(1e-6, max_depth, n_grid)
    
    # Pre-compute the fill probabilities for each depth
    # p_bid(d) = 1 - exp(-A * exp(-k * d) * dt)
    # p_ask(d) = 1 - exp(-A * exp(-k * d) * dt)
    exp_term = np.exp(-k * depth_grid)
    rate = A * exp_term * dt
    p_fill = 1.0 - np.exp(-rate)
    
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
            
            # No bid fill if q == Q
            # No ask fill if q == -Q
            
            if q == Q:
                # Only ask side available
                # Choose ask depth to maximize: -d * p_ask(d) + (1 - p_ask(d)) * V_next[q] + p_ask(d) * V_next[q+1]
                # But q+1 = Q+1 is out of bounds, so ask fill is not possible
                # Actually, re-reading: "no ask fill at q = -Q" means at q=-Q you can't sell (ask)
                # "no bid fill at q = Q" means at q=Q you can't buy (bid)
                # So at q=Q, bid is disabled, ask is enabled
                # At q=-Q, ask is disabled, bid is enabled
                
                # At q=Q: bid disabled, ask enabled
                # Ask fill: earn d, inventory becomes q-1 = Q-1
                # No fill: inventory stays Q
                # Value = -inv_cost + max_d [ -d*p_ask(d) + (1-p_ask(d))*V_next[q] + p_ask(d)*V_next[q-1] ]
                
                q_next_ask = q - 1  # Q-1
                q_next_ask_idx = q_next_ask + Q
                q_idx_next = q_idx  # q stays
                
                # Maximize over d
                # f(d) = -d * p_fill(d) + (1 - p_fill(d)) * V_next[q_idx] + p_fill(d) * V_next[q_next_ask_idx]
                values = -depth_grid * p_fill + (1 - p_fill) * V_next[q_idx] + p_fill * V_next[q_next_ask_idx]
                best_idx = np.argmax(values)
                ask_table[n, q_idx] = depth_grid[best_idx]
                V[q_idx] = -inv_cost + values[best_idx]
                
                # Bid is disabled
                bid_table[n, q_idx] = np.inf
                
            elif q == -Q:
                # Bid enabled, ask disabled
                # Bid fill: earn d, inventory becomes q+1 = -Q+1
                q_next_bid = q + 1  # -Q+1
                q_next_bid_idx = q_next_bid + Q
                q_idx_next = q_idx
                
                # Maximize over d
                values = -depth_grid * p_fill + (1 - p_fill) * V_next[q_idx] + p_fill * V_next[q_next_bid_idx]
                best_idx = np.argmax(values)
                bid_table[n, q_idx] = depth_grid[best_idx]
                V[q_idx] = -inv_cost + values[best_idx]
                
                ask_table[n, q_idx] = np.inf
                
            else:
                # Both sides available
                # Bid fill: earn d_b, inventory becomes q+1
                # Ask fill: earn d_a, inventory becomes q-1
                # No fill: inventory stays q
                # Fills are independent
                
                q_next_bid = q + 1
                q_next_ask = q - 1
                q_next_bid_idx = q_next_bid + Q
                q_next_ask_idx = q_next_ask + Q
                
                # For each pair (d_b, d_a), the value is:
                # -inv_cost + 
                #   p_b * p_a * (d_b + d_a + V_next[q]) +  # both fill: earn both, net inventory change 0
                #   p_b * (1-p_a) * (d_b + V_next[q+1]) +   # bid fills only
                #   (1-p_b) * p_a * (d_a + V_next[q-1]) +   # ask fills only
                #   (1-p_b) * (1-p_a) * V_next[q]
                
                # This is a 2D optimization which is expensive.
                # However, note that the objective is separable in some sense.
                # Let's expand:
                # = -inv_cost + 
                #   p_b*p_a*(d_b + d_a) + p_b*p_a*V_next[q] +
                #   p_b*(1-p_a)*d_b + p_b*(1-p_a)*V_next[q+1] +
                #   (1-p_b)*p_a*d_a + (1-p_b)*p_a*V_next[q-1] +
                #   (1-p_b)*(1-p_a)*V_next[q]
                
                # Group by d_b and d_a:
                # d_b terms: p_b*p_a*d_b + p_b*(1-p_a)*d_b = p_b*d_b*(p_a + 1 - p_a) = p_b*d_b
                # d_a terms: p_b*p_a*d_a + (1-p_b)*p_a*d_a = p_a*d_a*(p_b + 1 - p_b) = p_a*d_a
                
                # So the depth-dependent part is: p_b(d_b)*d_b + p_a(d_a)*d_a
                # And the inventory-dependent part is:
                # p_b*p_a*V_next[q] + p_b*(1-p_a)*V_next[q+1] + (1-p_b)*p_a*V_next[q-1] + (1-p_b)*(1-p_a)*V_next[q]
                
                # The depth choices d_b and d_a appear in both the "earnings" part and the "probability" part.
                # So it's NOT separable. We need to do a 2D search.
                
                # For efficiency, we can do a nested 1D search or a coarse 2D grid.
                # Let's use a coarser grid for the 2D search.
                n_grid_2d = 50
                depth_grid_2d = np.linspace(1e-6, max_depth, n_grid_2d)
                
                exp_term_2d = np.exp(-k * depth_grid_2d)
                rate_2d = A * exp_term_2d * dt
                p_fill_2d = 1.0 - np.exp(-rate_2d)
                
                # Create meshgrid
                D_b, D_a = np.meshgrid(depth_grid_2d, depth_grid_2d, indexing='ij')
                P_b = 1.0 - np.exp(-A * np.exp(-k * D_b) * dt)
                P_a = 1.0 - np.exp(-A * np.exp(-k * D_a) * dt)
                
                # Compute value for each (d_b, d_a)
                # Earnings: P_b * D_b + P_a * D_a
                # Inventory value: P_b*P_a*V_next[q] + P_b*(1-P_a)*V_next[q+1] + (1-P_b)*P_a*V_next[q-1] + (1-P_b)*(1-P_a)*V_next[q]
                earnings = P_b * D_b + P_a * D_a
                inv_val = (P_b * P_a * V_next[q_idx] + 
                          P_b * (1 - P_a) * V_next[q_next_bid_idx] + 
                          (1 - P_b) * P_a * V_next[q_next_ask_idx] + 
                          (1 - P_b) * (1 - P_a) * V_next[q_idx])
                
                total = earnings + inv_val
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
    bid_depths = bid_table[step, inv_indices]
    ask_depths = ask_table[step, inv_indices]
    
    # Ensure valid values
    bid_depths = np.where(bid_depths > -3.0 / params['k'], bid_depths, np.inf)
    ask_depths = np.where(ask_depths > -3.0 / params['k'], ask_depths, np.inf)
    
    return bid_depths, ask_depths
