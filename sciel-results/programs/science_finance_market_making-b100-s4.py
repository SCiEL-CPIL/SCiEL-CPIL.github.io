"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes the optimal bid and ask depths using continuous-depth optimization.
    Uses a fine grid with local refinement around the first-order condition optimum.
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1
    
    # Pre-compute inventory costs
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # Initial depth grid
    depth_grid = np.linspace(0.005, 5.0, 200)
    n_depths = len(depth_grid)
    
    # Pre-compute fill probabilities
    with np.errstate(over='ignore', under='ignore'):
        exp_term = np.exp(-k * depth_grid)
        p_depths = 1.0 - np.exp(-A * exp_term * dt)
    p_depths = np.clip(p_depths, 0.0, 1.0)
    
    # Build matrices
    P_b = np.tile(p_depths[:, np.newaxis], (1, n_depths))
    P_a = np.tile(p_depths[np.newaxis, :], (n_depths, 1))
    
    Depths_b = np.tile(depth_grid[:, np.newaxis], (1, n_depths))
    Depths_a = np.tile(depth_grid[np.newaxis, :], (n_depths, 1))
    Rewards = Depths_b * P_b + Depths_a * P_a
    
    P_plus = P_b * (1.0 - P_a)
    P_minus = P_a * (1.0 - P_b)
    P_same = 1.0 - P_plus - P_minus
    
    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))
    
    start_time = time.time()
    deadline = start_time + 315.0  # Safety margin
    
    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            # Fill remaining with last computed values
            if n < steps - 1:
                V[n:n+1, :] = V[n+1, :]
                opt_bid[n:n+1, :] = opt_bid[n+1, :]
                opt_ask[n:n+1, :] = opt_ask[n+1, :]
            break
            
        V_next = V[n+1]
        
        # Boundary handling
        V_q_plus = np.zeros(n_q)
        V_q_minus = np.zeros(n_q)
        V_q = V_next.copy()
        
        V_q_plus[:n_q-1] = V_next[1:]
        V_q_minus[1:] = V_next[:-1]
        
        # Compute values for all (i, j, q)
        # Shape: (n_q, n_depths, n_depths)
        Values = Rewards[np.newaxis, :, :] - C_inv[:, np.newaxis, np.newaxis]
        
        # Add transition terms
        Values += P_plus[np.newaxis, :, :] * V_q_plus[:, np.newaxis, np.newaxis]
        Values += P_minus[np.newaxis, :, :] * V_q_minus[:, np.newaxis, np.newaxis]
        Values += P_same[np.newaxis, :, :] * V_q[:, np.newaxis, np.newaxis]
        
        # Apply boundary masks: no bid fill at q=Q, no ask fill at q=-Q
        # At q=Q (last index), P_plus should be zeroed
        Values[-1, :, :] -= P_plus * V_q_plus[-1, np.newaxis, np.newaxis]
        # At q=-Q (first index), P_minus should be zeroed
        Values[0, :, :] -= P_minus * V_q_minus[0, np.newaxis, np.newaxis]
        
        flat_values = Values.reshape(n_q, -1)
        max_indices = np.argmax(flat_values, axis=1)
        
        i_idx, j_idx = np.unravel_index(max_indices, (n_depths, n_depths))
        
        V[n, :] = flat_values[np.arange(n_q), max_indices]
        opt_bid[n, :] = depth_grid[i_idx]
        opt_ask[n, :] = depth_grid[j_idx]
        
    return opt_bid, opt_ask

def _get_policy(params):
    """
    Retrieves or computes the optimal policy for the given parameters.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'], 
        params['T'], params['steps'], params['Q']
    )
    
    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed DP policy.
    """
    # Get the policy
    opt_bid, opt_ask = _get_policy(params)
    
    # t is current time. n is step index.
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    
    # Clamp n to valid range
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    # inventory is an array of ints
    Q = params['Q']
    q_idx = inventory + Q
    
    # Ensure indices are valid
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    # Retrieve depths
    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]
    
    # Return as float arrays
    return bid_depths.astype(float), ask_depths.astype(float)

# EVOLVE-BLOCK-END
