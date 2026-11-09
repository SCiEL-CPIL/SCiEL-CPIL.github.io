import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 290.0  # Safety margin
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
    M = 4000
    x_grid = np.logspace(-12, 0, M)
    log_x = np.log(x_grid)
    b_grid = -log_x / k
    p_grid = 1.0 - np.exp(-A * dt * x_grid)
    
    # Finer grid for polish
    M_fine = 8000
    x_grid_fine = np.logspace(-12, 0, M_fine)
    b_grid_fine = -np.log(x_grid_fine) / k
    p_grid_fine = 1.0 - np.exp(-A * dt * x_grid_fine)
    
    # V[t, q_index]
    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)
    
    # Precompute costs
    cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # Fixed point iterations per time step
    n_iter = 10
    
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
        
        # Polish pass: refine with finer grid around current solutions
        # For each q, find the index in fine grid closest to current b_curr/a_curr
        # and search locally
        
        # Only polish if we have time
        if time.time() < _DEADLINE - 30:
            for q_idx in range(n_q):
                # Polish bid
                if not np.isinf(b_curr[q_idx]):
                    # Find range in fine grid near b_curr[q_idx]
                    # b_grid_fine is sorted increasing
                    pos = np.searchsorted(b_grid_fine, b_curr[q_idx])
                    lo = max(0, pos - 50)
                    hi = min(M_fine, pos + 50)
                    
                    shift_b_q = dV_b[q_idx] + p_a_curr[q_idx] * D_cross[q_idx]
                    term_b_fine = shift_b_q + b_grid_fine[lo:hi]
                    val_b_fine = p_grid_fine[lo:hi] * term_b_fine
                    
                    if np.max(val_b_fine) > 0:
                        local_idx = np.argmax(val_b_fine)
                        b_curr[q_idx] = b_grid_fine[lo + local_idx]
                        p_b_curr[q_idx] = p_grid_fine[lo + local_idx]
                    
                    # Re-optimize ask with updated p_b
                    shift_a_q = dV_a[q_idx] + p_b_curr[q_idx] * D_cross[q_idx]
                    pos_a = np.searchsorted(b_grid_fine, a_curr[q_idx]) if not np.isinf(a_curr[q_idx]) else 0
                    if np.isinf(a_curr[q_idx]):
                        # Search full fine grid for ask
                        term_a_fine = shift_a_q + b_grid_fine
                        val_a_fine = p_grid_fine * term_a_fine
                        if np.max(val_a_fine) > 0:
                            local_idx_a = np.argmax(val_a_fine)
                            a_curr[q_idx] = b_grid_fine[local_idx_a]
                            p_a_curr[q_idx] = p_grid_fine[local_idx_a]
                    else:
                        lo_a = max(0, pos_a - 50)
                        hi_a = min(M_fine, pos_a + 50)
                        term_a_fine = shift_a_q + b_grid_fine[lo_a:hi_a]
                        val_a_fine = p_grid_fine[lo_a:hi_a] * term_a_fine
                        if np.max(val_a_fine) > 0:
                            local_idx_a = np.argmax(val_a_fine)
                            a_curr[q_idx] = b_grid_fine[lo_a + local_idx_a]
                            p_a_curr[q_idx] = p_grid_fine[lo_a + local_idx_a]
                else:
                    # b is inf, just optimize ask independently
                    shift_a_q = dV_a[q_idx]
                    if not np.isinf(a_curr[q_idx]):
                        pos_a = np.searchsorted(b_grid_fine, a_curr[q_idx])
                        lo_a = max(0, pos_a - 50)
                        hi_a = min(M_fine, pos_a + 50)
                        term_a_fine = shift_a_q + b_grid_fine[lo_a:hi_a]
                        val_a_fine = p_grid_fine[lo_a:hi_a] * term_a_fine
                        if np.max(val_a_fine) > 0:
                            local_idx_a = np.argmax(val_a_fine)
                            a_curr[q_idx] = b_grid_fine[lo_a + local_idx_a]
                            p_a_curr[q_idx] = p_grid_fine[lo_a + local_idx_a]
                
                # Re-optimize bid with updated p_a if b was finite
                if not np.isinf(b_curr[q_idx]):
                    shift_b_q = dV_b[q_idx] + p_a_curr[q_idx] * D_cross[q_idx]
                    pos = np.searchsorted(b_grid_fine, b_curr[q_idx])
                    lo = max(0, pos - 50)
                    hi = min(M_fine, pos + 50)
                    term_b_fine = shift_b_q + b_grid_fine[lo:hi]
                    val_b_fine = p_grid_fine[lo:hi] * term_b_fine
                    if np.max(val_b_fine) > 0:
                        local_idx = np.argmax(val_b_fine)
                        b_curr[q_idx] = b_grid_fine[lo + local_idx]
                        p_b_curr[q_idx] = p_grid_fine[lo + local_idx]
        
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
        
        # Recalculate to be sure
        contrib_b = np.zeros(n_q)
        contrib_a = np.zeros(n_q)
        
        valid_b = ~np.isinf(b_curr)
        valid_a = ~np.isinf(a_curr)
        
        contrib_b[valid_b] = p_b_curr[valid_b] * (b_curr[valid_b] + dV_b[valid_b] + p_a_curr[valid_b] * D_cross[valid_b])
        contrib_a[valid_a] = p_a_curr[valid_a] * (a_curr[valid_a] + dV_a[valid_a])
        
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
