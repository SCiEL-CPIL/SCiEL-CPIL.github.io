"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for pre-computed lookup tables per regime
_LUT_CACHE = {}

def _compute_lut(sigma, A, k, gamma, T, steps, Q):
    """
    Pre-compute optimal bid and ask depths for all time steps and inventory levels.
    Returns a dictionary with keys 'bid' and 'ask', each a 2D array of shape (steps, 2*Q+1).
    """
    dt = T / steps
    # Grid of candidate depths
    # The optimal depth is typically around 1/k. We scan from 0 to 5/k.
    # Using a fine grid ensures we capture the optimum.
    depth_grid = np.linspace(0.0, 5.0 / k, 200)
    
    # V_next[q_index] where q_index = q + Q
    V_next = np.zeros(2 * Q + 1)
    
    bid_lut = np.full((steps, 2 * Q + 1), np.inf)
    ask_lut = np.full((steps, 2 * Q + 1), np.inf)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(2 * Q + 1)
        
        for q in range(-Q, Q + 1):
            idx = q + Q
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # --- Optimize Bid ---
            if q < Q:
                # q_new = q - 1
                delta_v = V_next[idx - 1] - V_next[idx]
                # Maximize p(d) * (d + delta_v)
                # p(d) = 1 - exp(-A * exp(-k*d) * dt)
                # We can vectorize this over the depth grid
                p = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
                values = p * (depth_grid + delta_v)
                best_idx = np.argmax(values)
                bid_lut[n, idx] = depth_grid[best_idx]
                best_val_bid = values[best_idx] + V_next[idx]
            else:
                # Cannot bid
                bid_lut[n, idx] = np.inf
                best_val_bid = V_next[idx]
            
            # --- Optimize Ask ---
            if q > -Q:
                # q_new = q + 1
                delta_v = V_next[idx + 1] - V_next[idx]
                # Maximize p(d) * (d + delta_v)
                p = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
                values = p * (depth_grid + delta_v)
                best_idx = np.argmax(values)
                ask_lut[n, idx] = depth_grid[best_idx]
                best_val_ask = values[best_idx] + V_next[idx]
            else:
                # Cannot ask
                ask_lut[n, idx] = np.inf
                best_val_ask = V_next[idx]
            
            V_curr[idx] = best_val_bid + best_val_ask - cost
        
        V_next = V_curr
        
        # Optional: early termination check if time is tight, but 1000*61*200 is fast.
        # if time.time() > start_time + 20: break

    return {'bid': bid_lut, 'ask': ask_lut}

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Create a key for the regime
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = _compute_lut(sigma, A, k, gamma, T, steps, Q)
    
    luts = _LUT_CACHE[key]
    
    # Determine current step index
    # t = n * dt, so n = t / dt = t * steps / T
    n = int(round(t * steps / T))
    # Clamp n to valid range
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    # Map inventory to indices
    # inventory is in [-Q, Q]
    inv_indices = np.asarray(inventory) + Q
    
    # Retrieve depths
    bid_depths = luts['bid'][n, inv_indices]
    ask_depths = luts['ask'][n, inv_indices]
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
