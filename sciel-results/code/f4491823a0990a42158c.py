"""Market-making with inventory risk: Exact Dynamic Programming Solution."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_T0 = time.time()
_DEADLINE = 290.0  # Safety margin before 300s limit

# Cache for optimal depth tables per regime
# Key: (sigma, A, k, gamma, T, steps, Q)
_DP_CACHE = {}

def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the finite-horizon MDP using Dynamic Programming.
    Returns bid_depths and ask_depths tables of shape (steps, 2*Q+1).
    Indexing: q_idx = q + Q, where q in [-Q, Q].
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _DP_CACHE:
        return _DP_CACHE[key]

    dt = T / steps
    num_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # Value function V[t, q_idx]
    # V[steps, :] = 0 (no terminal charge)
    V_next = np.zeros(num_q)
    
    # Optimal depths
    bid_depths = np.zeros((steps, num_q))
    ask_depths = np.zeros((steps, num_q))
    
    # Pre-compute risk cost term: -0.5 * gamma * sigma^2 * q^2 * dt
    risk_cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # Time-homogeneous approximation for speed? 
    # No, the problem is finite horizon. The optimal policy changes with t.
    # However, for T=1 and steps=1000, the change is smooth.
    # We can solve backwards.
    
    # To speed up, we can observe that the optimal depth depends on the 
    # derivative of the value function w.r.t inventory.
    # But let's just do the grid search for depth at each state.
    # Grid for depth: [0, 5/k] with fine resolution.
    # 5/k is a reasonable upper bound.
    # Let's use 100 points for depth search.
    depth_grid = np.linspace(0, 5.0 / k, 100)
    
    # We need to be careful with time.
    # 1000 steps * 61 inventory states * 100 depth candidates = 6.1M evaluations.
    # Each evaluation involves exp and log. This might be slow in Python loops.
    # We need to vectorize the depth search for all q at a given t.
    
    # Let's iterate backwards in time.
    for t in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fallback: use current best or simple heuristic
            # If we run out of time, fill remaining with a heuristic
            # based on the last computed step or default
            for t_rem in range(t, -1, -1):
                if t_rem < steps - 1:
                    bid_depths[t_rem] = bid_depths[t_rem + 1]
                    ask_depths[t_rem] = ask_depths[t_rem + 1]
            break
        
        # For each inventory level q, find optimal bid_depth and ask_depth
        # Objective for state (t, q):
        # R = p_bid * bid_d + p_ask * ask_d + risk_cost[q] + 
        #     (1 - p_bid - p_ask) * V_next[q] + 
        #     p_bid * V_next[q-1] + p_ask * V_next[q+1]
        # Note: V_next is indexed by q_idx.
        # q-1 corresponds to q_idx - 1, q+1 to q_idx + 1.
        
        # We can vectorize over q for a fixed depth candidate?
        # Or vectorize over depth candidates for all q?
        
        # Let's vectorize over q for each depth candidate.
        # But bid and ask depths are chosen independently?
        # Yes, given the structure, we can optimize bid_depth and ask_depth separately?
        # No, they are coupled through the probability of no fill?
        # Actually, the term (1 - p_bid - p_ask) * V_next[q] couples them.
        # However, p_bid and p_ask are small.
        # Let's check if we can decouple.
        # Maximize over d_bid, d_ask:
        # p_b(d_b) d_b + p_a(d_a) d_a + C + (1 - p_b(d_b) - p_a(d_a)) V[q] 
        # + p_b(d_b) V[q-1] + p_a(d_a) V[q+1]
        
        # Let f_b(d_b) = p_b(d_b) (d_b + V[q-1] - V[q])
        # Let f_a(d_a) = p_a(d_a) (d_a + V[q+1] - V[q])
        # Then total = f_b(d_b) + f_a(d_a) + C + V[q]
        # Since the terms are separable in d_b and d_a, we can optimize them independently!
        
        # Optimal d_b maximizes: p_b(d) * (d + V[q-1] - V[q])
        # Optimal d_a maximizes: p_a(d) * (d + V[q+1] - V[q])
        
        # This is a 1D optimization for each q.
        
        V_curr = np.zeros(num_q)
        
        # V[q-1] for q in [-Q, Q]
        # q = -Q -> q-1 = -Q-1 (out of bounds, but p_bid=0 at q=Q, p_ask=0 at q=-Q)
        # Wait, at q=-Q, p_ask=0. So d_ask doesn't matter.
        # At q=Q, p_bid=0. So d_bid doesn't matter.
        
        # Let's compute the optimal depths for all q simultaneously using the grid.
        
        # For each q_idx, we need V[q-1] and V[q+1].
        # V_next is the array for t+1.
        
        # Handle boundaries for V lookups
        # V_prev = V_next[q-1], V_next_q = V_next[q], V_next_q_plus = V_next[q+1]
        
        # Create padded V_next to handle boundaries easily?
        # Or just use np.roll? No, boundaries are absorbing/reflecting in a way?
        # No, at q=Q, no bid fill. So we don't transition to Q+1.
        # So we only need V_next[q-1] for q < Q.
        
        # Let's compute the objective function for each depth candidate for all q.
        
        # d_grid: (100,)
        # q_idx: (61,)
        
        # We want to find argmax_d for each q.
        
        # Let's compute p_b(d) and p_a(d) for all d in grid, for all q?
        # p_b depends on q? No, p_b depends on d and t (via dt, A, k).
        # Wait, p_b = 1 - exp(-A exp(-k d) dt). It does NOT depend on q directly, 
        # except that we don't quote if q=Q.
        
        # So p_b(d) is the same for all q (where q < Q).
        # p_a(d) is the same for all q (where q > -Q).
        
        # This is a huge simplification!
        # The optimal depth for bid at state (t, q) depends on (V[q-1] - V[q]).
        # The optimal depth for ask at state (t, q) depends on (V[q+1] - V[q]).
        
        # Let delta_V_minus = V_next[q-1] - V_next[q]
        # Let delta_V_plus = V_next[q+1] - V_next[q]
        
        # Maximize p(d) * (d + delta_V)
        
        # We can pre-compute p(d) for the grid.
        p_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
        
        # For each q, we need to find d in depth_grid that maximizes p(d) * (d + delta_V)
        
        # Let's vectorize:
        # Create a matrix M of shape (num_q, 100)
        # M[i, j] = p_grid[j] * (depth_grid[j] + delta_V[i])
        
        # delta_V_minus for q in [-Q, Q]
        # q_idx 0 (q=-Q): p_bid=0, so doesn't matter. Set delta=0.
        # q_idx 1 (q=-Q+1): V[q-1] is V[-Q]
        # ...
        
        # Let's construct delta_V_minus and delta_V_plus arrays.
        
        # V_next is indexed 0..2Q
        # q_idx i corresponds to q = i - Q
        
        # delta_V_minus[i] = V_next[i-1] - V_next[i] for i > 0
        # delta_V_minus[0] = 0 (boundary)
        
        # delta_V_plus[i] = V_next[i+1] - V_next[i] for i < 2Q
        # delta_V_plus[2Q] = 0 (boundary)
        
        dvm = np.zeros(num_q)
        dvm[1:] = V_next[:-1] - V_next[1:]
        
        dvp = np.zeros(num_q)
        dvp[:-1] = V_next[1:] - V_next[:-1]
        
        # Now, for each q_idx, find optimal d_bid and d_ask.
        
        # Bid:
        # Objective: p_grid * (depth_grid + dvm[None, :])
        # Note: dvm is (num_q,). We need to broadcast.
        # M_bid = p_grid[None, :] * (depth_grid[None, :] + dvm[:, None])
        
        # This creates a (num_q, 100) array.
        # Find argmax along axis 1.
        
        # But wait, if q=Q, p_bid=0, so we should set depth to 0 or inf?
        # The task says "np.inf means no quote".
        # If we set depth to 0, p_bid = 1 - exp(-A dt) > 0.
        # But the model says "no bid fill at q=Q".
        # So if q=Q, we must return inf for bid_depth.
        # Similarly, if q=-Q, return inf for ask_depth.
        
        # Let's compute the optimal depths first, then handle boundaries.
        
        # To avoid large memory usage, we can do this in chunks or just let numpy handle it.
        # 61 * 100 = 6100 floats. Trivial.
        
        M_bid = p_grid[None, :] * (depth_grid[None, :] + dvm[:, None])
        idx_bid = np.argmax(M_bid, axis=1)
        opt_bid = depth_grid[idx_bid]
        
        M_ask = p_grid[None, :] * (depth_grid[None, :] + dvp[:, None])
        idx_ask = np.argmax(M_ask, axis=1)
        opt_ask = depth_grid[idx_ask]
        
        # Apply boundary conditions
        # q_idx 2Q (q=Q): bid_depth = inf
        # q_idx 0 (q=-Q): ask_depth = inf
        
        if Q > 0:
            opt_bid[2 * Q] = np.inf
            opt_ask[0] = np.inf
        
        bid_depths[t] = opt_bid
        ask_depths[t] = opt_ask
        
        # Update V_curr
        # V[t, q] = max_obj + risk_cost[q]
        # max_obj for bid = M_bid[i, idx_bid[i]]
        # max_obj for ask = M_ask[i, idx_ask[i]]
        # Total max reward from fills = max_obj_bid + max_obj_ask
        
        # Wait, is it sum?
        # R = p_b d_b + p_a d_a + ...
        # Yes, the terms are additive.
        
        max_reward_bid = M_bid[np.arange(num_q), idx_bid]
        max_reward_ask = M_ask[np.arange(num_q), idx_ask]
        
        # Add the base value V_next[q] and risk cost
        # V[t, q] = max_reward_bid + max_reward_ask + V_next[q] + risk_cost[q]
        
        V_curr = max_reward_bid + max_reward_ask + V_next + risk_cost
        
        V_next = V_curr
    
    _DP_CACHE[key] = (bid_depths, ask_depths)
    return _DP_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Quote policy using pre-computed DP tables.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        # Fallback: simple heuristic
        k = params['k']
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Get optimized tables
    bid_depths_table, ask_depths_table = _solve_dp(sigma, A, k, gamma, T, steps, Q)
    
    # Map t to step index
    # t = n * dt
    # n = t / dt
    dt = T / steps
    n = int(round(t / dt))
    
    # Clamp n to valid range
    n = np.clip(n, 0, steps - 1)
    
    # Get depths for current step
    # inventory is an array of ints
    # q_idx = inventory + Q
    q_idx = inventory + Q
    
    # Handle potential out of bounds?
    # inventory is guaranteed to be in [-Q, Q]
    
    bid_depths = bid_depths_table[n, q_idx]
    ask_depths = ask_depths_table[n, q_idx]
    
    # Ensure valid depths
    # Values below -3/k are invalid. Our grid is [0, 5/k], so we are safe.
    # np.inf is allowed for no quote.
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
