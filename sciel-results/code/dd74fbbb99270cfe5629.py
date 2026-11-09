import numpy as np
import time

_cache = {}
_start_time = time.time()
DEADLINE = 310.0  # 330s wall clock, leave 20s margin

def _solve_regime(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
    
    # Check deadline
    if time.time() > DEADLINE:
        return None
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # Grid for x = exp(-k*b), so b = -ln(x)/k
    # x in (0, 1], b in [0, inf)
    # For large b, fill prob ~ 1 - exp(-A*dt*exp(-k*b)) ~ A*dt*exp(-k*b) ~ A*dt*x
    # The optimal b balances: marginal benefit of deeper quote (lower fill prob) vs lower depth earned
    # g(x) = p(x) * (C - ln(x)/k) where p(x) = 1 - exp(-lambda_t * x)
    # lambda_t = A * dt
    
    lambda_t = A * dt
    
    # Choose grid resolution
    # b_max: for b > ~50/k, x < exp(-50) ~ 2e-22, p(x) ~ lambda_t * x, g ~ lambda_t * x * (C - ln(x)/k) -> 0
    # So b_max ~ 50/k is sufficient
    # For k=0.75, b_max ~ 67. For k=1.5, b_max ~ 33.
    b_max = 60.0 / k
    n_grid = 2000
    
    # Log-spaced x from exp(-b_max*k) to 1
    x_grid = np.exp(-k * np.linspace(0, b_max, n_grid))
    log_x = np.log(x_grid)  # = -k * linspace(0, b_max)
    p_grid = 1.0 - np.exp(-lambda_t * x_grid)
    
    # V[t, i] = optimal value from step t with inventory q_i
    V = np.zeros((N + 1, n_q), dtype=np.float64)
    opt_b = np.full((N, n_q), np.inf, dtype=np.float64)
    opt_a = np.full((N, n_q), np.inf, dtype=np.float64)
    
    inv_k = 1.0 / k
    
    for t in range(N - 1, -1, -1):
        if time.time() > DEADLINE:
            # Return what we have so far; remaining steps use zero depths
            _cache[key] = (opt_b, opt_a)
            return _cache[key]
        
        V_next = V[t + 1]
        
        # C_b[i] = V_next[i+1] - V_next[i] for i < 2Q (q < Q, can have bid fill)
        # C_a[i] = V_next[i-1] - V_next[i] for i > 0 (q > -Q, can have ask fill)
        C_b = np.zeros(n_q, dtype=np.float64)
        C_a = np.zeros(n_q, dtype=np.float64)
        
        # For q < Q (idx 0..2Q-1): bid fill increases inventory
        C_b[:2*Q] = V_next[1:] - V_next[:-1]
        # C_b[2Q] stays 0 (no bid at q=Q)
        
        # For q > -Q (idx 1..2Q): ask fill decreases inventory
        C_a[1:] = V_next[:-1] - V_next[1:]
        # C_a[0] stays 0 (no ask at q=-Q)
        
        # Optimize bid depth:
        # g_b(x) = p(x) * (C_b - ln(x)/k)
        # We want max over x in grid, or x=0 (b=inf, no quote, value 0)
        
        term_b = C_b[:, None] - inv_k * log_x[None, :]  # shape (n_q, n_grid)
        g_b = p_grid[None, :] * term_b  # shape (n_q, n_grid)
        
        max_idx_b = np.argmax(g_b, axis=1)
        max_val_b = np.max(g_b, axis=1)
        
        b_vals = -log_x[max_idx_b] * inv_k  # = b at optimal x
        # If max_val_b <= 0, don't quote
        b_vals[max_val_b <= 1e-12] = np.inf
        # Force no bid at q=Q
        b_vals[2*Q] = np.inf
        
        opt_b[t, :] = b_vals
        
        # Optimize ask depth:
        term_a = C_a[:, None] - inv_k * log_x[None, :]
        g_a = p_grid[None, :] * term_a
        
        max_idx_a = np.argmax(g_a, axis=1)
        max_val_a = np.max(g_a, axis=1)
        
        a_vals = -log_x[max_idx_a] * inv_k
        a_vals[max_val_a <= 1e-12] = np.inf
        # Force no ask at q=-Q
        a_vals[0] = np.inf
        
        opt_a[t, :] = a_vals
        
        # Update V
        contrib_b = np.where(np.isinf(b_vals), 0.0, max_val_b)
        contrib_a = np.where(np.isinf(a_vals), 0.0, max_val_a)
        
        V[t, :] = V_next + cost + contrib_b + contrib_a
    
    _cache[key] = (opt_b, opt_a)
    return _cache[key]

def quote(t, inventory, mid, params):
    result = _solve_regime(params)
    
    if result is None:
        # Deadline exceeded, return no quotes
        n = len(inventory)
        return np.full(n, np.inf, dtype=np.float64), np.full(n, np.inf, dtype=np.float64)
    
    opt_b, opt_a = result
    dt = params['T'] / params['steps']
    Q = params['Q']
    
    t_idx = int(round(t / dt))
    if t_idx < 0:
        t_idx = 0
    if t_idx >= opt_b.shape[0]:
        t_idx = opt_b.shape[0] - 1
    
    idxs = inventory + Q
    
    bid_depths = opt_b[t_idx, idxs].astype(np.float64)
    ask_depths = opt_a[t_idx, idxs].astype(np.float64)
    
    # Ensure valid: no negative, no inf below -3/k (shouldn't happen), no NaN
    # Depths must be >= 0. Clip any negative to 0 (shouldn't occur)
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    # Replace any NaN with inf
    bid_depths = np.where(np.isnan(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), np.inf, ask_depths)
    
    return bid_depths, ask_depths
