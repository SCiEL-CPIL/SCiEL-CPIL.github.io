"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}

def _compute_policy(sigma, A, k, gamma, T, steps, Q, depth_grid):
    """
    Computes the optimal bid and ask depths for each (time, inventory) state.
    Uses vectorized DP with a refined depth grid and boundary handling.
    """
    dt = T / steps
    depths = depth_grid
    n_depths = len(depths)
    
    # State space: q ranges from -Q to Q
    # V[n, q_idx] where q_idx = q + Q
    V = np.zeros((steps + 1, 2 * Q + 1))
    opt_bid = np.zeros((steps, 2 * Q + 1))
    opt_ask = np.zeros((steps, 2 * Q + 1))
    
    # Pre-compute fill probabilities for each depth
    # p(d) = 1 - exp(-A * exp(-k * d) * dt)
    # Handle potential overflow/underflow gracefully
    with np.errstate(over='ignore', under='ignore'):
        exp_term = np.exp(-k * depths)
        p_depths = 1.0 - np.exp(-A * exp_term * dt)
    # Clip to [0, 1] for numerical stability
    p_depths = np.clip(p_depths, 0.0, 1.0)
    
    # Pre-compute meshgrids for rewards and probabilities
    # p_b[i, j] = p_depths[i] (bid depth is i)
    # p_a[i, j] = p_depths[j] (ask depth is j)
    P_b = np.tile(p_depths[:, np.newaxis], (1, n_depths))
    P_a = np.tile(p_depths[np.newaxis, :], (n_depths, 1))
    
    # Rewards: b * p_b + a * p_a
    Depths_b = np.tile(depths[:, np.newaxis], (1, n_depths))
    Depths_a = np.tile(depths[np.newaxis, :], (n_depths, 1))
    Rewards = Depths_b * P_b + Depths_a * P_a
    
    # Probability matrices for state transitions
    # P_plus[i, j] = p_b[i] * (1 - p_a[j]) -> inventory increases by 1
    # P_minus[i, j] = p_a[j] * (1 - p_b[i]) -> inventory decreases by 1
    # P_same[i, j] = 1 - P_plus - P_minus
    P_plus = P_b * (1.0 - P_a)
    P_minus = P_a * (1.0 - P_b)
    P_same = 1.0 - P_plus - P_minus
    
    # Pre-compute inventory costs
    q_vals = np.arange(-Q, Q + 1)
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    start_time = time.time()
    timeout = 320.0 # Safety margin for 330s limit
    
    for n in range(steps - 1, -1, -1):
        if time.time() - start_time > timeout:
            # If we run out of time, break. The rest of opt_bid/ask will be 0.
            # This is a safety fallback.
            break
            
        V_next = V[n+1]
        
        # Vectorize over all q indices
        # We need to compute max over (i, j) of:
        # Rewards[i, j] - C_inv[q] + P_plus[i, j]*V_next[q+1] + P_minus[i, j]*V_next[q-1] + P_same[i, j]*V_next[q]
        
        # V_next is shape (2Q+1,)
        # We want an array of shape (2Q+1, n_depths, n_depths)
        
        V_q_plus = np.zeros((2*Q+1, 1, 1))
        V_q_minus = np.zeros((2*Q+1, 1, 1))
        V_q = np.zeros((2*Q+1, 1, 1))
        
        # Fill V_q
        V_q[:, 0, 0] = V_next
        
        # Fill V_q_plus
        # For q_idx 0..2Q-1, V_q_plus[q_idx] = V_next[q_idx+1]
        V_q_plus[:2*Q, 0, 0] = V_next[1:]
        # For q_idx 2Q (q=Q), no bid fill, so this term is multiplied by 0 in P_plus?
        # Actually, if q=Q, P_plus should be 0.
        # In our P_plus matrix, it's non-zero.
        # So we must handle boundaries.
        # Easier: set V_q_plus[2Q, 0, 0] = 0.0 and rely on P_plus being zeroed?
        # No, P_plus is precomputed and non-zero.
        # We need to zero out P_plus for q=Q and P_minus for q=-Q.
        
        # Fill V_q_minus
        # For q_idx 1..2Q, V_q_minus[q_idx] = V_next[q_idx-1]
        V_q_minus[1:, 0, 0] = V_next[:-1]
        
        # Let's create boundary masks.
        # mask_plus: shape (2Q+1, 1, 1), 1 where q < Q (q_idx < 2Q), 0 where q == Q (q_idx == 2Q)
        mask_plus = np.ones((2*Q+1, 1, 1))
        mask_plus[-1, 0, 0] = 0.0
        
        # mask_minus: shape (2Q+1, 1, 1), 1 where q > -Q (q_idx > 0), 0 where q == -Q (q_idx == 0)
        mask_minus = np.ones((2*Q+1, 1, 1))
        mask_minus[0, 0, 0] = 0.0
        
        # Adjust P_plus and P_minus
        # P_plus is (n_depths, n_depths). We need it to be (2Q+1, n_depths, n_depths) with masks.
        # Or we can just multiply the terms.
        
        # E_V_next = P_plus * V_q_plus * mask_plus + P_minus * V_q_minus * mask_minus + P_same * V_q
        # Note: P_same needs adjustment too?
        # If q=Q, p_b=0.
        # Original P_same = (1-p_b)(1-p_a) + p_b p_a = 1 - p_a + p_b p_a - p_b + p_b p_a? No.
        # P_same = 1 - p_b(1-p_a) - p_a(1-p_b) = 1 - p_b + p_b p_a - p_a + p_a p_b = 1 - p_b - p_a + 2 p_b p_a.
        # If p_b=0, P_same = 1 - p_a.
        # Our formula P_same = 1 - P_plus - P_minus.
        # If we zero P_plus for q=Q, P_same becomes 1 - 0 - P_minus = 1 - p_a. Correct.
        # If we zero P_minus for q=-Q, P_same becomes 1 - P_plus - 0 = 1 - p_b. Correct.
        
        # So we can just zero out the P terms.
        P_plus_masked = P_plus[np.newaxis, :, :] * mask_plus
        P_minus_masked = P_minus[np.newaxis, :, :] * mask_minus
        P_same_masked = (1.0 - P_plus_masked - P_minus_masked)
        
        # Now compute total value
        # Shape: (2Q+1, n_depths, n_depths)
        Values = Rewards[np.newaxis, :, :] - C_inv[:, np.newaxis, np.newaxis] + \
                 P_plus_masked * V_q_plus + \
                 P_minus_masked * V_q_minus + \
                 P_same_masked * V_q
        
        # Find max over last two axes
        # argmax returns indices in (n_depths, n_depths) space
        # We need to flatten the last two axes to find global max per q
        flat_values = Values.reshape(2*Q+1, -1)
        max_indices = np.argmax(flat_values, axis=1)
        
        # Unravel indices
        i_idx, j_idx = np.unravel_index(max_indices, (n_depths, n_depths))
        
        V[n, :] = flat_values[np.arange(2*Q+1), max_indices]
        opt_bid[n, :] = depths[i_idx]
        opt_ask[n, :] = depths[j_idx]
        
    return opt_bid, opt_ask

def _get_policy(params):
    """
    Retrieves or computes the optimal policy for the given parameters.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    # Depth grid
    # Use a log-uniform or piecewise grid for better resolution near small depths?
    # Or just a finer linear grid.
    # Let's try 80 points from 0.005 to 4.0
    depth_grid = np.linspace(0.005, 4.0, 80)
    
    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'], 
        params['T'], params['steps'], params['Q'], depth_grid
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
    # t = n * dt => n = t / dt
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    
    # Clamp n to valid range
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    # inventory is an array of ints
    # We need to map inventory q to index q_idx = q + Q
    Q = params['Q']
    q_idx = inventory + Q
    
    # Ensure indices are valid (though inventory should be in [-Q, Q])
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    # Retrieve depths
    # opt_bid[n, q_idx]
    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]
    
    # Return as float arrays
    return bid_depths.astype(float), ask_depths.astype(float)

# EVOLVE-BLOCK-END
