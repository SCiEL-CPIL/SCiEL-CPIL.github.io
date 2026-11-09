import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 285.0  # Safety margin
_cache = {}

def _solve_regime(params):
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
    N = steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # Grid for x = exp(-k * depth)
    # x in [1e-12, 1]
    M = 2000
    x_grid = np.logspace(-12, 0, M)
    log_x = np.log(x_grid)
    b_grid = -log_x / k
    p_grid = 1.0 - np.exp(-A * dt * x_grid)
    
    # V[t, q_index]
    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)
    
    # Precompute costs
    cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # Fixed point iterations per time step
    n_iter = 8
    
    for t in range(N - 1, -1, -1):
        V_next = V[t + 1]
        
        # Initialize p_b, p_a to 0 (or small values)
        # We will iterate to find optimal b, a
        # Start with decoupled optimum as initial guess
        
        # 1. Initial guess: Decoupled
        # Delta V_b[q] = V[q+1] - V[q]
        # Delta V_a[q] = V[q-1] - V[q]
        
        dV_b = np.zeros(n_q)
        dV_a = np.zeros(n_q)
        if n_q > 1:
            dV_b[:-1] = V_next[1:] - V_next[:-1]
            dV_a[1:] = V_next[:-1] - V_next[1:]
        
        # D_cross[q] = V[q+1] + V[q-1] - 2V[q]
        D_cross = np.zeros(n_q)
        if n_q > 2:
            D_cross[1:-1] = V_next[2:] + V_next[:-2] - 2 * V_next[1:-1]
        # Boundaries:
        # q=Q: D_cross not used for bid (b=inf), but for ask?
        # If q=Q, ask moves to Q-1. Bid is inf.
        # The cross term only matters if both can fill.
        # If q=Q, p_b=0, so p_b p_a D_cross = 0.
        
        # We will store current optimal p_b, p_a
        p_b_curr = np.zeros(n_q)
        p_a_curr = np.zeros(n_q)
        b_curr = np.full(n_q, np.inf)
        a_curr = np.full(n_q, np.inf)
        
        # Iteration loop
        for _ in range(n_iter):
            # Update shifts based on current p's
            # Effective bid reward: b + dV_b + p_a_curr * D_cross
            shift_b = dV_b + p_a_curr * D_cross
            # Effective ask reward: a + dV_a + p_b_curr * D_cross
            shift_a = dV_a + p_b_curr * D_cross
            
            # Optimize b for each q
            # Maximize p_grid[i] * (b_grid[i] + shift_b[q])
            # Matrix: M x n_q
            # term_b[q, i] = b_grid[i] + shift_b[q]
            # val_b[q, i] = p_grid[i] * term_b[q, i]
            
            # Broadcasting: shift_b[:, None] (n_q, 1), b_grid[None, :] (1, M)
            term_b = shift_b[:, None] + b_grid[None, :]
            val_b = p_grid[None, :] * term_b
            
            # Handle boundaries: q=Q (index 2Q) -> b=inf
            # We can just set the max val to -inf for that row before argmax, or mask later.
            # Masking is cleaner.
            # Actually, if we just compute, the optimal b for q=Q might be finite, but we force it to inf.
            
            max_idx_b = np.argmax(val_b, axis=1)
            max_val_b = np.max(val_b, axis=1)
            
            b_new = b_grid[max_idx_b]
            p_b_new = p_grid[max_idx_b]
            
            # Force boundaries
            b_new[2 * Q] = np.inf
            p_b_new[2 * Q] = 0.0
            
            # If max_val is very small, it might be better to not quote?
            # The problem says "returning... distances... np.inf means no quote".
            # If max_val_b < 0, it's better to set b=inf?
            # The objective is sum of rewards. If the expected contribution is negative, we should avoid the fill.
            # However, the "no quote" option corresponds to b=inf (p=0).
            # The grid includes x=1 (b=0).
            # If all values in val_b row are negative, argmax picks the least negative.
            # But we can choose p=0 (b=inf).
            # So, if max_val_b < 0, set b=inf, p=0.
            
            neg_mask_b = max_val_b < 0
            b_new[neg_mask_b] = np.inf
            p_b_new[neg_mask_b] = 0.0
            
            # Optimize a for each q
            term_a = shift_a[:, None] + b_grid[None, :]
            val_a = p_grid[None, :] * term_a
            
            max_idx_a = np.argmax(val_a, axis=1)
            max_val_a = np.max(val_a, axis=1)
            
            a_new = b_grid[max_idx_a] # b_grid is same as a_grid
            p_a_new = p_grid[max_idx_a]
            
            # Force boundaries: q=-Q (index 0) -> a=inf
            a_new[0] = np.inf
            p_a_new[0] = 0.0
            
            neg_mask_a = max_val_a < 0
            a_new[neg_mask_a] = np.inf
            p_a_new[neg_mask_a] = 0.0
            
            # Check for convergence or just take the last iteration
            b_curr = b_new
            a_curr = a_new
            p_b_curr = p_b_new
            p_a_curr = p_a_new
        
        # Newton polish for finite depths (where p > 0 and depth is finite)
        # For bid: maximize f(b) = p(b) * (b + shift_b[q])
        # f'(b) = p'(b)*(b+s) + p(b) = -k*x*(b+s) + (1-x)
        # where x = exp(-k*b), s = shift_b[q]
        # f'(b) = 0 => (1-x) = k*x*(b+s) => 1 = x*(1 + k*(b+s))
        # Newton: b_{n+1} = b_n - f'/f''
        
        # Polish bid depths
        finite_b_mask = np.isfinite(b_curr)
        if np.any(finite_b_mask):
            q_idx = np.where(finite_b_mask)[0]
            b_val = b_curr[q_idx].copy()
            s_b = shift_b[q_idx]
            
            for _newton in range(3):
                x = np.exp(-k * b_val)
                p = 1.0 - x
                # f = p * (b + s)
                # f' = p'(b+s) + p = -k*x*(b+s) + p
                f_prime = -k * x * (b_val + s_b) + p
                # f'' = d/db[-k*x*(b+s) + p] = -k*(-k*x)*(b+s) + -k*x + (-k*x) = k^2*x*(b+s) - 2*k*x
                f_double_prime = k**2 * x * (b_val + s_b) - 2 * k * x
                # Newton step
                valid = np.abs(f_double_prime) > 1e-15
                b_val[valid] -= f_prime[valid] / f_double_prime[valid]
                # Clamp to reasonable range
                b_val = np.clip(b_val, 1e-6, 12.0 / k)
            
            b_curr[q_idx] = b_val
        
        # Polish ask depths
        finite_a_mask = np.isfinite(a_curr)
        if np.any(finite_a_mask):
            q_idx = np.where(finite_a_mask)[0]
            a_val = a_curr[q_idx].copy()
            s_a = shift_a[q_idx]
            
            for _newton in range(3):
                x = np.exp(-k * a_val)
                p = 1.0 - x
                f_prime = -k * x * (a_val + s_a) + p
                f_double_prime = k**2 * x * (a_val + s_a) - 2 * k * x
                valid = np.abs(f_double_prime) > 1e-15
                a_val[valid] -= f_prime[valid] / f_double_prime[valid]
                a_val = np.clip(a_val, 1e-6, 12.0 / k)
            
            a_curr[q_idx] = a_val
        
        # Store optimal actions for this t
        opt_b[t, :] = b_curr
        opt_a[t, :] = a_curr
        
        # Update V[t]
        # V[t, q] = V_next[q] + cost[q] + 
        #            p_b(q) * (b(q) + dV_b(q) + p_a(q) * D_cross(q)) + 
        #            p_a(q) * (a(q) + dV_a(q)) 
        # Wait, the decomposition was:
        # J = p_b(b) + p_a(a) + p_b(dV_b) + p_a(dV_a) + p_b p_a D_cross
        #   = p_b (b + dV_b + p_a D_cross) + p_a (a + dV_a)
        # This is exactly what we maximized in the last iteration (shift_a used p_b_curr, shift_b used p_a_curr).
        # So the max value is:
        # max_val_b_final = p_b_curr * (b_curr + dV_b + p_a_curr * D_cross)
        # max_val_a_final = p_a_curr * (a_curr + dV_a)
        
        # Recalculate with polished values
        # Update p values after polish
        p_b_polished = np.zeros(n_q)
        finite_b = np.isfinite(b_curr)
        if np.any(finite_b):
            p_b_polished[finite_b] = 1.0 - np.exp(-k * b_curr[finite_b])
        
        p_a_polished = np.zeros(n_q)
        finite_a = np.isfinite(a_curr)
        if np.any(finite_a):
            p_a_polished[finite_a] = 1.0 - np.exp(-k * a_curr[finite_a])
        
        contrib_b = p_b_polished * (b_curr + dV_b + p_a_polished * D_cross)
        contrib_b = np.where(np.isinf(b_curr), 0.0, contrib_b)
        
        contrib_a = p_a_polished * (a_curr + dV_a)
        contrib_a = np.where(np.isinf(a_curr), 0.0, contrib_a)
        
        V[t, :] = V_next + cost + contrib_b + contrib_a
        
    _cache[key] = (opt_b, opt_a)
    return _cache[key]

def quote(t, inventory, mid, params):
    if time.time() > _DEADLINE:
        return np.full(4000, np.inf), np.full(4000, np.inf)
        
    opt_b, opt_a = _solve_regime(params)
    dt = params['T'] / params['steps']
    Q = params['Q']
    
    t_idx = int(t / dt)
    if t_idx >= len(opt_b):
        t_idx = len(opt_b) - 1
        
    idxs = inventory + Q
    
    bid_depths = opt_b[t_idx, idxs]
    ask_depths = opt_a[t_idx, idxs]
    
    # Ensure validity
    # Values below -3/k are invalid. Our b_grid starts at -log(1e-12)/k = 12/k > 0.
    # So all values are > 0.
    # NaN check
    bid_depths = np.where(np.isnan(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), np.inf, ask_depths)
    
    return bid_depths, ask_depths
