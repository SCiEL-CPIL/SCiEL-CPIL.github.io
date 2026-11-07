import numpy as np
import time

# Cache for precomputed tables
_cache = {}

def _get_tables(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
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
    
    N_states = 2 * Q + 1
    V_next = np.zeros(N_states)
    
    B_table = np.full((steps, N_states), np.inf)
    A_table = np.full((steps, N_states), np.inf)
    
    # Grid for u = exp(-k * depth)
    # u in [1e-6, 1]
    # 100 points is good resolution
    u_grid = np.logspace(-6, 0, 100)
    depths_grid = -1.0 / k * np.log(u_grid)
    p_grid = 1.0 - np.exp(-A * u_grid * dt)
    
    start_time = time.time()
    
    for n in range(steps - 1, -1, -1):
        # Vectorized over q
        q_indices = np.arange(N_states) - Q
        
        # Deltas
        # delta_v_plus[i] = V_next[i+1] - V_next[i] for q < Q, else 0
        delta_v_plus = np.zeros(N_states)
        delta_v_plus[:-1] = V_next[1:] - V_next[:-1]
        
        # delta_v_minus[i] = V_next[i] - V_next[i-1] for q > -Q, else 0
        delta_v_minus = np.zeros(N_states)
        delta_v_minus[1:] = V_next[1:] - V_next[:-1]
        
        cost = 0.5 * gamma * sigma**2 * (q_indices**2) * dt
        
        # C_b = dt * delta_v_minus
        C_b = dt * delta_v_minus
        # C_a = -dt * delta_v_plus
        C_a = -dt * delta_v_plus
        
        # Val_b shape (N_states, 100)
        # depths_grid[None, :] * p_grid[None, :] - C_b[:, None] * p_grid[None, :]
        val_b = depths_grid[None, :] * p_grid[None, :] - C_b[:, None] * p_grid[None, :]
        val_a = depths_grid[None, :] * p_grid[None, :] - C_a[:, None] * p_grid[None, :]
        
        idx_max_b = np.argmax(val_b, axis=1)
        idx_max_a = np.argmax(val_a, axis=1)
        
        b_opts = depths_grid[idx_max_b]
        a_opts = depths_grid[idx_max_a]
        
        max_b_part = val_b[np.arange(N_states), idx_max_b]
        max_a_part = val_a[np.arange(N_states), idx_max_a]
        
        # Apply bounds
        # q == Q -> index Q+Q = 2Q
        # q == -Q -> index 0
        b_opts[2*Q] = np.inf
        a_opts[0] = np.inf
        
        # If q=Q, max_b_part should be 0?
        # If we set b=inf, p_b=0. Reward 0. Cost 0.
        # Our val_b calculation assumed we could choose u.
        # At q=Q, we force b=inf. The value contribution from b is 0.
        # The term dt * delta_v_minus * p_b is also 0.
        # So max_b_part should be 0 for q=Q.
        max_b_part[2*Q] = 0.0
        max_a_part[0] = 0.0
        
        V_curr = max_b_part + max_a_part - cost + dt * V_next
        
        B_table[n, :] = b_opts
        A_table[n, :] = a_opts
        V_next = V_curr
        
        # Time check
        if time.time() - start_time > 250:
             # Fallback to simple strategy if precomputation is too slow
             # Just return 1/k for all
             B_table[:] = 1.0 / k
             A_table[:] = 1.0 / k
             break
             
    _cache[key] = (B_table, A_table)
    return _cache[key]

def quote(t, inventory, mid, params):
    B_table, A_table = _get_tables(params)
    
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Determine step index
    n = int(round(t / dt))
    n = np.clip(n, 0, steps - 1)
    
    # Map inventory to index
    # inventory is in [-Q, Q]
    q_indices = inventory + Q
    
    bid_depths = B_table[n, q_indices]
    ask_depths = A_table[n, q_indices]
    
    return bid_depths, ask_depths
