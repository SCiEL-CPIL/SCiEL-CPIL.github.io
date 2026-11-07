"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0

_OPT_TABLES = {}

def _get_optimal_table(params):
    """
    Pre-compute optimal bid and ask depths for all (step, inventory) states
    using backward induction on the discrete HJB equation with a reduced depth grid.
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["Q"], params["steps"])
    if key in _OPT_TABLES:
        return _OPT_TABLES[key]
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    
    # Reduced depth grid: use quantiles of exponential distribution
    # Depths range from very small to large; we'll use a log-spaced grid
    # The optimal depth is roughly around 1/k, so we focus there
    n_depths = 200
    # Use a grid that covers [0.01, 10/k] with more resolution near 1/k
    depth_min = 0.01
    depth_max = 10.0 / k
    depths = np.geomspace(depth_min, depth_max, n_depths)
    
    # Pre-compute fill probabilities for each depth
    # p_fill = 1 - exp(-A * exp(-k * depth) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)  # shape (n_depths,)
    
    # State space: inventory in [-Q, Q], step in [0, steps-1]
    n_inv = 2 * Q + 1
    inv_indices = np.arange(-Q, Q + 1)  # shape (n_inv,)
    
    # Value function: V[step, inv_idx] = optimal expected value from step to end
    # We'll store value and optimal depths
    # V shape: (steps, n_inv)
    # opt_bid_depth_idx shape: (steps, n_inv)
    # opt_ask_depth_idx shape: (steps, n_inv)
    
    # Initialize value at terminal step to 0
    V = np.zeros((steps + 1, n_inv))
    opt_bid_idx = np.zeros((steps, n_inv), dtype=np.int32)
    opt_ask_idx = np.zeros((steps, n_inv), dtype=np.int32)
    
    # Pre-compute holding costs for all inventory levels
    holding_cost = 0.5 * gamma * sigma**2 * inv_indices**2 * dt  # shape (n_inv,)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 10.0:
            # Fallback: use uniform depths
            fallback_depth = 1.0 / k
            for i in range(n_inv):
                opt_bid_idx[n, i] = np.argmin(np.abs(depths - fallback_depth))
                opt_ask_idx[n, i] = np.argmin(np.abs(depths - fallback_depth))
            continue
        
        # For each inventory level, compute optimal bid and ask depths
        for i in range(n_inv):
            V_next_i = V[n+1, i]
            
            # Bid side
            if i < n_inv - 1:  # can receive bid fill (q < Q)
                V_next_bid = V[n+1, i+1]
                # Expected value from bid depth d:
                # p(d) * (d + V_next_bid) + (1-p(d)) * V_next_i
                # = p(d) * d + V_next_i + p(d) * (V_next_bid - V_next_i)
                bid_vals = fill_probs * depths + V_next_i + fill_probs * (V_next_bid - V_next_i)
                best_bid_i = np.argmax(bid_vals)
            else:
                best_bid_i = np.argmax(fill_probs * depths)  # no inventory change benefit
            
            # Ask side
            if i > 0:  # can receive ask fill (q > -Q)
                V_next_ask = V[n+1, i-1]
                ask_vals = fill_probs * depths + V_next_i + fill_probs * (V_next_ask - V_next_i)
                best_ask_i = np.argmax(ask_vals)
            else:
                best_ask_i = np.argmax(fill_probs * depths)
            
            opt_bid_idx[n, i] = best_bid_i
            opt_ask_idx[n, i] = best_ask_i
            
            # Compute V[n, i]
            p_bid = fill_probs[best_bid_i]
            p_ask = fill_probs[best_ask_i]
            
            EV_bid = 0.0
            EV_ask = 0.0
            EV_none = 0.0
            
            if i < n_inv - 1:
                EV_bid = p_bid * (depths[best_bid_i] + V[n+1, i+1])
            if i > 0:
                EV_ask = p_ask * (depths[best_ask_i] + V[n+1, i-1])
            EV_none = (1.0 - p_bid) * (1.0 - p_ask) * V_next_i
            
            V[n, i] = -holding_cost[i] + EV_bid + EV_ask + EV_none
    
    # Build lookup tables for actual depths
    bid_depths_table = depths[opt_bid_idx]  # shape (steps, n_inv)
    ask_depths_table = depths[opt_ask_idx]  # shape (steps, n_inv)
    
    result = {
        'bid': bid_depths_table,
        'ask': ask_depths_table,
        'depths': depths,
        'n_inv': n_inv,
    }
    _OPT_TABLES[key] = result
    return result


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths using pre-computed HJB optimal policy.
    Returns (bid_depths, ask_depths) each of shape (len(inventory),).
    """
    table = _get_optimal_table(params)
    
    Q = params["Q"]
    steps = params["steps"]
    
    # Determine current step index
    dt = params["T"] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Map inventory to index: inv_idx = q + Q
    inv_idx = inventory + Q
    # Clamp to valid range
    inv_idx = np.clip(inv_idx, 0, table['n_inv'] - 1)
    
    # Look up depths
    bid_depths = table['bid'][n, inv_idx]
    ask_depths = table['ask'][n, inv_idx]
    
    # Ensure depths are valid: must be > -3/k, not -inf, not NaN
    k = params["k"]
    min_depth = -3.0 / k + 1e-10  # just above the invalid threshold
    
    # Replace any invalid values with a safe default
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_depth), bid_depths, 1.0/k)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_depth), ask_depths, 1.0/k)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
