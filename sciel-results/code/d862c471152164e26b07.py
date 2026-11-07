"""
Strategy: Pre-compute the optimal policy using Dynamic Programming (Bellman Equation)
with vectorized inner loop and higher resolution grid.
Includes a time guard to ensure completion within the 330s limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for pre-computed optimal policies
_policy_cache = {}
_start_time = time.time()
TIME_LIMIT = 320.0  # Safety margin

def _compute_optimal_policy(params):
    """
    Compute the optimal bid and ask depths for all (t, q) states using DP.
    Vectorized over inventory states for speed.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)
    
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))
    
    V_next = np.zeros(n_states)
    
    # Grid resolution
    max_depth = 3.0
    n_grid = 300
    grid = np.linspace(0, max_depth, n_grid)
    
    # Precompute probabilities
    exp_term = np.exp(-k * grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    one_minus_p = 1.0 - p_grid
    
    # Precompute cost for each state
    cost_per_state = 0.5 * gamma * sigma**2 * q_values**2 * dt  # (n_states,)
    
    # Precompute broadcast arrays
    grid_b = grid[None, :, None]   # (1, n_grid, 1)
    grid_a = grid[None, None, :]   # (1, 1, n_grid)
    pb_col = p_grid[None, :, None]         # (1, n_grid, 1)
    pa_row = p_grid[None, None, :]         # (1, 1, n_grid)
    ompb_col = one_minus_p[None, :, None]  # (1, n_grid, 1)
    ompa_row = one_minus_p[None, None, :]  # (1, 1, n_grid)
    
    # We will iterate t from steps-1 down to 0
    for t in range(steps - 1, -1, -1):
        # Check time
        if time.time() - _start_time > TIME_LIMIT:
            break
            
        # v_q, v_qp, v_qm for all states
        v_q = V_next  # (n_states,)
        
        # v_qp: V_next[i+1] for i < n_states-1, else 0 (no bid at q=Q)
        v_qp = np.zeros(n_states)
        v_qp[:-1] = V_next[1:]
        
        # v_qm: V_next[i-1] for i > 0, else 0 (no ask at q=-Q)
        v_qm = np.zeros(n_states)
        v_qm[1:] = V_next[:-1]
        
        # Build 3D total: (n_states, n_grid, n_grid)
        # t1: no fills
        t1 = v_q[:, None, None] * ompb_col * ompa_row
        
        # t2: bid fill only
        t2 = v_qp[:, None, None] * pb_col * ompa_row + grid_b * pb_col * ompa_row
        
        # t3: ask fill only
        t3 = v_qm[:, None, None] * ompb_col * pa_row + grid_a * ompb_col * pa_row
        
        # t4: both fills
        t4 = v_q[:, None, None] * pb_col * pa_row + (grid_b + grid_a) * pb_col * pa_row
        
        total = t1 + t2 + t3 + t4 - cost_per_state[:, None, None]
        
        # Find argmax over last two axes for each state
        total_flat = total.reshape(n_states, -1)
        max_idx = np.argmax(total_flat, axis=1)
        i = max_idx // n_grid
        j = max_idx % n_grid
        
        opt_bid[t, :] = grid[i]
        opt_ask[t, :] = grid[j]
        V_next = total_flat[np.arange(n_states), max_idx]
        
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """
    Look up optimal depths from the pre-computed policy.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _policy_cache[key]
    
    n = int(np.round(t * params['steps']))
    if n >= params['steps']:
        n = params['steps'] - 1
        
    indices = inventory + params['Q']
    
    bids = bid_depths[n, indices]
    asks = ask_depths[n, indices]
    
    return bids, asks

# EVOLVE-BLOCK-END
