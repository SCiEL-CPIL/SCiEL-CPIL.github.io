import numpy as np

_cache = {}

def _compute_table(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    risk_coeff = 0.5 * gamma * sigma**2 * dt
    
    # Grid
    # 1/k is the peak of d*exp(-kd).
    # Skew might shift it.
    # Use 0 to 6/k to be safe.
    d_max = 6.0 / k
    num_points = 100
    d_grid = np.linspace(0, d_max, num_points)
    
    exp_kd = np.exp(-k * d_grid)
    p_vec = 1 - np.exp(-A * exp_kd * dt)
    
    p_b_full = np.broadcast_to(p_vec[:, None], (num_points, num_points))
    p_a_full = np.broadcast_to(p_vec[None, :], (num_points, num_points))
    d_b_full = np.broadcast_to(d_grid[:, None], (num_points, num_points))
    d_a_full = np.broadcast_to(d_grid[None, :], (num_points, num_points))
    
    reward_grid = p_b_full * d_b_full + p_a_full * d_a_full
    
    V_next = np.zeros(2 * Q + 1)
    opt_table = np.zeros((steps, 2 * Q + 1, 2))
    
    for n in range(steps - 1, -1, -1):
        V_curr = V_next
        
        V_n = np.zeros(2 * Q + 1)
        
        for idx in range(2 * Q + 1):
            q = idx - Q
            
            W_q = V_curr[idx]
            
            if q < Q:
                W_q_plus = V_curr[idx + 1]
            else:
                W_q_plus = 0.0
                
            if q > -Q:
                W_q_minus = V_curr[idx - 1]
            else:
                W_q_minus = 0.0
                
            if q == Q:
                p_b_eff = np.zeros_like(p_b_full)
            else:
                p_b_eff = p_b_full
                
            if q == -Q:
                p_a_eff = np.zeros_like(p_a_full)
            else:
                p_a_eff = p_a_full
                
            p_none = (1 - p_b_eff) * (1 - p_a_eff)
            p_b_only = p_b_eff * (1 - p_a_eff)
            p_a_only = (1 - p_b_eff) * p_a_eff
            p_bb = p_b_eff * p_a_eff
            
            E_V = (p_none + p_bb) * W_q + p_b_only * W_q_plus + p_a_only * W_q_minus
            
            risk = risk_coeff * q**2
            F = reward_grid - risk + E_V
            
            max_val = np.max(F)
            max_idx = np.unravel_index(np.argmax(F), F.shape)
            
            V_n[idx] = max_val
            opt_table[n, idx, 0] = d_grid[max_idx[0]]
            opt_table[n, idx, 1] = d_grid[max_idx[1]]
            
        V_next = V_n
        
    return opt_table

def quote(t, inventory, mid, params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _cache:
        _cache[key] = _compute_table(params)
        
    table = _cache[key]
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    indices = inventory + Q
    
    bid_depths = table[n, indices, 0]
    ask_depths = table[n, indices, 1]
    
    return bid_depths, ask_depths
