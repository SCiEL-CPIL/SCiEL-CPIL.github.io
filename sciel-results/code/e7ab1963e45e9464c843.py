"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 325.0  # 5s safety margin before 330s kill
_cache = {}

def _get_dp_tables(params):
    """Compute optimal bid/ask depth tables via dynamic programming with vectorized grid search."""
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
    # V[steps][q] = 0 (no terminal charge)
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Optimal depth tables
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)
    
    # Precompute risk penalty
    q_indices = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma**2 * q_indices**2 * dt
    
    # Precompute depth grid for vectorized search
    # Use a fine grid for depth search
    d_max = 15.0 / k  # reasonable upper bound
    n_grid = 500
    depth_grid = np.linspace(0.0, d_max, n_grid)
    
    # Precompute fill probabilities for the grid
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    p_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    def max_reward_vectorized(delta_V, q_idx, is_bid):
        """Maximize p(d) * (d + delta_V) over d >= 0 using precomputed grid."""
        if (is_bid and q_idx >= Q) or (not is_bid and q_idx <= -Q):
            return 0.0, np.inf
        
        # For each depth in grid, compute reward = p(d) * (d + delta_V)
        rewards = p_grid * (depth_grid + delta_V)
        
        # Find max
        idx_max = np.argmax(rewards)
        d_opt = depth_grid[idx_max]
        reward = rewards[idx_max]
        
        # Refine with local search around the best grid point
        if 0 < idx_max < n_grid - 1:
            # Local quadratic interpolation for better estimate
            i = idx_max
            d0, d1, d2 = depth_grid[i-1], depth_grid[i], depth_grid[i+1]
            r0, r1, r2 = rewards[i-1], rewards[i], rewards[i+1]
            
            # Fit quadratic: r = a*d^2 + b*d + c
            # Use Lagrange interpolation
            denom = (d0 - d1) * (d0 - d2) * (d1 - d2)
            if abs(denom) > 1e-15:
                a = (r0 / ((d0-d1)*(d0-d2)) + r1 / ((d1-d0)*(d1-d2)) + r2 / ((d2-d0)*(d2-d1)))
                b = -(r0*(d0+d1) / ((d0-d1)*(d0-d2)) + r1*(d1+d0) / ((d1-d0)*(d1-d2)) + r2*(d2+d0) / ((d2-d0)*(d2-d1)))
                
                if a < 0:  # concave
                    d_star = -b / (2 * a)
                    if d0 < d_star < d2:
                        p_star = 1.0 - np.exp(-A * np.exp(-k * d_star) * dt)
                        r_star = p_star * (d_star + delta_V)
                        if r_star > r1:
                            d_opt = d_star
                            reward = r_star
        
        return reward, d_opt
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > TIME_LIMIT - 30:
            # Not enough time; fill remaining with current best estimates
            break
        
        for q_idx in range(-Q, Q + 1):
            V_next_q = V[n + 1, q_idx + Q]
            
            # Bid side
            if q_idx < Q:
                delta_b = V[n + 1, q_idx + 1 + Q] - V_next_q
                bid_reward, bid_depth = max_reward_vectorized(delta_b, q_idx, True)
            else:
                bid_reward = 0.0
                bid_depth = np.inf
            
            # Ask side
            if q_idx > -Q:
                delta_a = V[n + 1, q_idx - 1 + Q] - V_next_q
                ask_reward, ask_depth = max_reward_vectorized(delta_a, q_idx, False)
            else:
                ask_reward = 0.0
                ask_depth = np.inf
            
            V[n, q_idx + Q] = V_next_q - risk[q_idx + Q] + bid_reward + ask_reward
            opt_bid[n, q_idx + Q] = bid_depth
            opt_ask[n, q_idx + Q] = ask_depth
    
    # If we broke out early, fill remaining steps with the last computed step
    # (or with a simple heuristic)
    for n in range(steps - 1, -1, -1):
        if np.all(np.isinf(opt_bid[n])):
            if n + 1 < steps:
                opt_bid[n] = opt_bid[n + 1]
                opt_ask[n] = opt_ask[n + 1]
            else:
                # Fallback: use 1/k
                base = 1.0 / k
                opt_bid[n] = np.where(q_indices < Q, base, np.inf)
                opt_ask[n] = np.where(q_indices > -Q, base, np.inf)
    
    _cache[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """Quote optimal depths based on dynamic programming solution."""
    # Deadline guard
    if time.time() - _START_TIME > TIME_LIMIT - 5:
        k = params['k']
        base = 1.0 / k
        N = len(inventory)
        bid_depths = np.full(N, base)
        ask_depths = np.full(N, base)
        # Handle boundaries
        Q = params['Q']
        bid_depths[inventory == Q] = np.inf
        ask_depths[inventory == -Q] = np.inf
        return bid_depths.astype(np.float64), ask_depths.astype(np.float64)
    
    opt_bid, opt_ask = _get_dp_tables(params)
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    # Current step index
    n = int(t / dt)
    n = min(n, steps - 1)  # clamp to valid range
    
    # Map inventory to indices
    inv_idx = inventory + Q
    
    bid_depths = opt_bid[n, inv_idx]
    ask_depths = opt_ask[n, inv_idx]
    
    # Safety: ensure valid values
    min_depth = -3.0 / params['k'] + 0.01
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_depth), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_depth), ask_depths, np.inf)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)

# EVOLVE-BLOCK-END
