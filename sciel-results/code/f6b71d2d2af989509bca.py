"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def _solve_depth_newton(A, k, dt, delta, d0, max_iter=50, tol=1e-14):
    """
    Solve the FOC for optimal depth d given inventory change delta.
    f(d) = p(d) * (d + delta) where p(d) = 1 - exp(-A*exp(-k*d)*dt).
    FOC: p'(d)*(d+delta) + p(d) = 0
    Uses Newton's method with careful handling of edge cases.
    """
    d = max(0.0, min(d0, 15.0 / k))
    
    for _ in range(max_iter):
        exp_kd = np.exp(-k * d)
        lam = A * exp_kd * dt
        
        if lam < 1e-10:
            # p ≈ lam, p' ≈ k*lam
            p = lam
            p_prime = k * lam
            p_double_prime = -k * k * lam
        else:
            exp_neg_lam = np.exp(-lam)
            p = 1.0 - exp_neg_lam
            p_prime = k * lam * exp_neg_lam
            p_double_prime = -k * k * lam * exp_neg_lam * (1.0 - lam)
        
        g = p_prime * (d + delta) + p
        g_prime = p_double_prime * (d + delta) + p_prime
        
        if abs(g_prime) < 1e-30:
            break
            
        newton_step = g / g_prime
        d_new = d - newton_step
        
        if d_new < 0:
            d_new = 0
        elif d_new > 15.0 / k:
            d_new = 15.0 / k
        
        if abs(d_new - d) < tol:
            d = d_new
            break
        d = d_new
    
    # Final check: if d is very small and delta is negative, check if 0 is better
    # Also verify the solution is actually a maximum by checking nearby points
    if d > 0:
        # Evaluate at d, d*0.5, d*1.5, 0 to ensure we're at a max
        candidates = [0.0, d]
        if d > 1e-6:
            candidates.append(d * 0.5)
            candidates.append(d * 1.5)
        
        best_val = -np.inf
        best_d = d
        for dc in candidates:
            dc = max(0.0, min(dc, 15.0 / k))
            exp_kd = np.exp(-k * dc)
            lam = A * exp_kd * dt
            if lam < 1e-10:
                p = lam
            else:
                p = 1.0 - np.exp(-lam)
            val = p * (dc + delta)
            if val > best_val:
                best_val = val
                best_d = dc
        d = best_d
    
    return d

def _get_dp_tables(params):
    """Compute optimal bid/ask depth tables via dynamic programming with Newton-refined FOC solution."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
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
    
    # Value table: V[n, q] for n=0..steps, q=-Q..Q
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Optimal depth tables
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)
    
    # Precompute risk penalty
    q_indices = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma**2 * q_indices**2 * dt
    
    # Grid for initial guesses - use more points for better resolution
    d_max = 15.0 / k
    num_points = 800
    D_grid = np.linspace(0.0, d_max, num_points)
    
    # Precompute fill probabilities for the grid
    exp_term = np.exp(-k * D_grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    def _optimal_depth(delta_val):
        """Find optimal depth for a given delta using grid search + Newton refinement."""
        # Quick check: if delta is very large positive, optimal depth is likely 0
        # If delta is very negative, we need to be more careful
        
        vals = p_grid * (D_grid + delta_val)
        max_val = np.max(vals)
        
        if max_val <= 0:
            return np.inf, 0.0
        
        idx = np.argmax(vals)
        d0 = D_grid[idx]
        
        # Newton refinement
        d_opt = _solve_depth_newton(A, k, dt, delta_val, d0)
        
        # Evaluate at the Newton solution
        exp_kd = np.exp(-k * d_opt)
        lam = A * exp_kd * dt
        if lam < 1e-10:
            p_opt = lam
        else:
            p_opt = 1.0 - np.exp(-lam)
        
        f_opt = p_opt * (d_opt + delta_val)
        
        # Also evaluate at nearby grid points to catch any issues
        if idx > 0:
            f_left = p_grid[idx-1] * (D_grid[idx-1] + delta_val)
            if f_left > f_opt:
                d_opt = D_grid[idx-1]
                f_opt = f_left
        if idx < num_points - 1:
            f_right = p_grid[idx+1] * (D_grid[idx+1] + delta_val)
            if f_right > f_opt:
                d_opt = D_grid[idx+1]
                f_opt = f_right
        
        # Safety: if Newton went wrong, fall back to grid best
        if f_opt < max_val - 1e-10:
            d_opt = d0
            f_opt = max_val
        
        return d_opt, max(f_opt, 0.0)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            # Fallback: use current V and no quotes for remaining steps
            break
        
        V_next = V[n + 1]
        
        # Calculate Delta V for bid and ask
        delta_b = np.zeros(2 * Q + 1)
        if Q > 0:
            delta_b[:-1] = V_next[1:] - V_next[:-1]
            # delta_b[-1] remains 0 (no bid at Q)
        
        delta_a = np.zeros(2 * Q + 1)
        if Q > 0:
            delta_a[1:] = V_next[:-1] - V_next[1:]
            # delta_a[0] remains 0 (no ask at -Q)
        
        bid_depths = np.full(2 * Q + 1, np.inf)
        ask_depths = np.full(2 * Q + 1, np.inf)
        bid_rewards = np.zeros(2 * Q + 1)
        ask_rewards = np.zeros(2 * Q + 1)
        
        # Vectorized optimization: group by similar delta values
        # For efficiency, process in batches
        bid_deltas = delta_b[:2*Q]  # Exclude boundary
        ask_deltas = delta_a[1:]    # Exclude boundary
        
        # Use a simpler approach: evaluate on a fine grid and pick best
        # This avoids Newton method overhead for each individual delta
        
        # For bid: maximize p(d) * (d + delta_b[q]) for each q
        # For ask: maximize p(d) * (d + delta_a[q]) for each q
        
        # Since deltas vary smoothly with q, we can use the grid directly
        # and interpolate
        
        # Create evaluation matrix: for each delta, compute objective on D_grid
        # bid_obj[i, j] = p_grid[j] * (D_grid[j] + bid_deltas[i])
        bid_obj = p_grid[np.newaxis, :] * (D_grid[np.newaxis, :] + bid_deltas[:, np.newaxis])
        bid_best_idx = np.argmax(bid_obj, axis=1)
        bid_best_val = bid_obj[np.arange(len(bid_deltas)), bid_best_idx]
        
        # Ask side
        ask_obj = p_grid[np.newaxis, :] * (D_grid[np.newaxis, :] + ask_deltas[:, np.newaxis])
        ask_best_idx = np.argmax(ask_obj, axis=1)
        ask_best_val = ask_obj[np.arange(len(ask_deltas)), ask_best_idx]
        
        # Now refine with Newton for each
        for i in range(2 * Q):
            if bid_best_val[i] > 0:
                d0 = D_grid[bid_best_idx[i]]
                d_opt = _solve_depth_newton(A, k, dt, bid_deltas[i], d0)
                exp_kd = np.exp(-k * d_opt)
                lam = A * exp_kd * dt
                if lam < 1e-10:
                    p_opt = lam
                else:
                    p_opt = 1.0 - np.exp(-lam)
                f_opt = p_opt * (d_opt + bid_deltas[i])
                
                if f_opt >= bid_best_val[i] - 1e-10:
                    bid_depths[i] = d_opt
                    bid_rewards[i] = max(f_opt, 0.0)
                else:
                    bid_depths[i] = d0
                    bid_rewards[i] = max(bid_best_val[i], 0.0)
            else:
                bid_depths[i] = np.inf
                bid_rewards[i] = 0.0
        
        for i in range(2 * Q):
            if ask_best_val[i] > 0:
                d0 = D_grid[ask_best_idx[i]]
                d_opt = _solve_depth_newton(A, k, dt, ask_deltas[i], d0)
                exp_kd = np.exp(-k * d_opt)
                lam = A * exp_kd * dt
                if lam < 1e-10:
                    p_opt = lam
                else:
                    p_opt = 1.0 - np.exp(-lam)
                f_opt = p_opt * (d_opt + ask_deltas[i])
                
                if f_opt >= ask_best_val[i] - 1e-10:
                    ask_depths[i + 1] = d_opt
                    ask_rewards[i + 1] = max(f_opt, 0.0)
                else:
                    ask_depths[i + 1] = d0
                    ask_rewards[i + 1] = max(ask_best_val[i], 0.0)
            else:
                ask_depths[i + 1] = np.inf
                ask_rewards[i + 1] = 0.0
        
        # Handle boundary conditions explicitly
        if Q > 0:
            bid_depths[-1] = np.inf
            bid_rewards[-1] = 0.0
            ask_depths[0] = np.inf
            ask_rewards[0] = 0.0
        
        # Update Value
        V[n] = V_next - risk + bid_rewards + ask_rewards
        opt_bid[n] = bid_depths
        opt_ask[n] = ask_depths
    
    _cache[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """Quote optimal depths based on dynamic programming solution."""
    # Deadline guard
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        n_paths = len(inventory)
        return np.full(n_paths, np.inf), np.full(n_paths, np.inf)
    
    opt_bid, opt_ask = _get_dp_tables(params)
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    # Current step index
    n = int(round(t / dt))
    n = min(max(n, 0), steps - 1)
    
    # Map inventory to indices
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    bid_depths = opt_bid[n, inv_idx]
    ask_depths = opt_ask[n, inv_idx]
    
    # Safety: ensure valid values
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 0), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 0), ask_depths, np.inf)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)

# EVOLVE-BLOCK-END
