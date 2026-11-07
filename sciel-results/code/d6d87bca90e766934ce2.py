"""Optimal market making with inventory risk using pre-computed backward induction."""
import numpy as np
import time

_POLICY_CACHE = {}


def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths via backward induction on the discrete model.
    
    Returns (bid_table, ask_table) each of shape (steps, 2*Q+1).
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_range = np.arange(-Q, Q + 1)  # -Q, -Q+1, ..., Q
    inv_index = inv_range + Q  # 0, 1, ..., 2Q
    
    # Value function V[q] for current step
    # V[q] = optimal expected objective from current step onwards given inventory q
    V = np.zeros(n_inv)
    
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # Pre-compute depth grid for optimization
    # Depths are positive distances from mid
    # We need to optimize over bid_depth > 0 and ask_depth > 0
    # The fill probability is 1 - exp(-A * exp(-k * depth) * dt)
    # For small dt, this is approximately A * exp(-k * depth) * dt
    
    # Use a logarithmic grid for depths to cover the relevant range
    # depth = 1/k is the "risk-neutral" depth where A*exp(-k*depth)*dt = A*dt/k
    # Relevant range: from very small (high fill prob) to large (low fill prob)
    # Let's use depths from 0.01 to 10/k
    max_depth = 10.0 / k
    n_depth = 200
    depth_grid = np.linspace(0.01, max_depth, n_depth)
    
    # Pre-compute fill probabilities for the depth grid
    # p_bid(d) = 1 - exp(-A * exp(-k * d) * dt)
    exp_term = np.exp(-k * depth_grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    for n in range(steps - 1, -1, -1):
        time_remaining = (steps - n) * dt
        # Inventory cost coefficient: gamma/2 * sigma^2 * q^2 * dt
        # This is the cost for being in inventory q during this step
        
        # For each inventory level, find optimal bid and ask depths
        for qi in range(n_inv):
            q = inv_range[qi]
            
            # Inventory cost for this step
            inv_cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Try to find optimal bid depth
            # If q == Q, no bid fill allowed (bid_depth = inf)
            if q >= Q:
                bid_table[n, qi] = np.inf
                ask_table[n, qi] = np.inf  # placeholder, will be set below
            else:
                # Optimize bid depth
                # Expected value from quoting bid at depth d:
                # p(d) * [d + V[q+1]] + (1-p(d)) * V[q] - inv_cost
                # = p(d) * (d + V[q+1] - V[q]) - inv_cost
                # Maximize over d > 0
                
                # The term (d + V[q+1] - V[q]) is the "gain" from a bid fill
                # Let delta_V_bid = V[q+1] - V[q]
                delta_V_bid = V[qi + 1] - V[qi] if qi + 1 < n_inv else 0.0
                
                # Maximize p(d) * (d + delta_V_bid) over d
                # p(d) = 1 - exp(-A * exp(-k*d) * dt)
                # For small dt, p(d) ≈ A * exp(-k*d) * dt
                # So we maximize A*dt*exp(-k*d) * (d + delta_V_bid)
                # Take derivative w.r.t. d:
                # A*dt * [-k*exp(-k*d)*(d + delta_V_bid) + exp(-k*d)] = 0
                # => -k*(d + delta_V_bid) + 1 = 0
                # => d* = (1 - k*delta_V_bid) / k = 1/k - delta_V_bid
                
                # But we need to be more careful with the exact p(d)
                # Let's use the pre-computed grid and find the max
                
                gains = depth_grid + delta_V_bid
                # Clip gains to be positive (if delta_V_bid is very negative, some depths give negative gain)
                # Actually, if gains are negative, we don't want to quote
                obj = p_grid * gains
                idx = np.argmax(obj)
                best_obj = obj[idx]
                
                # Check if it's worth quoting at all
                # If best_obj <= 0, don't quote (inf depth)
                if best_obj > 0:
                    bid_table[n, qi] = depth_grid[idx]
                else:
                    bid_table[n, qi] = np.inf
            
            # Optimize ask depth
            if q <= -Q:
                ask_table[n, qi] = np.inf
            else:
                # For ask fill, inventory decreases by 1
                delta_V_ask = V[qi - 1] - V[qi] if qi - 1 >= 0 else 0.0
                
                gains = depth_grid + delta_V_ask
                obj = p_grid * gains
                idx = np.argmax(obj)
                best_obj = obj[idx]
                
                if best_obj > 0:
                    ask_table[n, qi] = depth_grid[idx]
                else:
                    ask_table[n, qi] = np.inf
            
            # Update V[q] = max over bid/ask combinations... 
            # Actually, bid and ask are independent, so we can compute them separately
            # V[q] = -inv_cost + max_bid + max_ask where max is over the choice to quote or not
            
            # Recompute max values
            if q >= Q:
                bid_val = 0.0
            else:
                delta_V_bid = V[qi + 1] - V[qi] if qi + 1 < n_inv else 0.0
                gains = depth_grid + delta_V_bid
                obj = p_grid * gains
                bid_val = max(0.0, np.max(obj))
            
            if q <= -Q:
                ask_val = 0.0
            else:
                delta_V_ask = V[qi - 1] - V[qi] if qi - 1 >= 0 else 0.0
                gains = depth_grid + delta_V_ask
                obj = p_grid * gains
                ask_val = max(0.0, np.max(obj))
            
            V[qi] = -inv_cost + bid_val + ask_val
    
    return bid_table, ask_table


def _get_policy_table(params):
    """Get or compute the policy table for the given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy_table(
            params['sigma'], params['A'], params['k'], params['gamma'], 
            params['T'], params['steps'], params['Q']
        )
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    
    Args:
        t: current time
        inventory: array of shape (4000,) with integer inventory levels
        mid: array of shape (4000,) with mid prices
        params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        (bid_depths, ask_depths) each of shape (4000,)
    """
    Q = params['Q']
    steps = params['steps']
    
    # Get step index
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Get policy table
    bid_table, ask_table = _get_policy_table(params)
    
    # Convert inventory to index
    # inventory is in [-Q, Q], index = inventory + Q
    inv_idx = np.clip(inventory + Q, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[n, inv_idx]
    ask_depths = ask_table[n, inv_idx]
    
    # Ensure no bid quote at q=Q, no ask quote at q=-Q
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    # Validate: depths must be > -3/k, not -inf, not NaN
    min_depth = -3.0 / params['k']
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_depth), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_depth), ask_depths, np.inf)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)
