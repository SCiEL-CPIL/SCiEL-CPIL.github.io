"""
Optimal market making via DP with vectorized fine grid search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 320.0


def _heuristic_depths(params, t, inventory):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    q = inventory.astype(np.float64)
    t_remaining = np.maximum(T - t, 0.0)
    d0 = 1.0 / k
    skew_scale = gamma * sigma * sigma * t_remaining
    alpha = skew_scale / (A * k) if A > 0 and k > 0 else 0.0
    alpha = min(alpha, 6.0 / k)
    bid_depth = d0 + alpha * q
    ask_depth = d0 - alpha * q
    min_depth = 0.01 / k
    max_depth = 10.0 / k
    bid_depth = np.clip(bid_depth, min_depth, max_depth)
    ask_depth = np.clip(ask_depth, min_depth, max_depth)
    bid_depth = np.where(inventory >= Q, np.inf, bid_depth)
    ask_depth = np.where(inventory <= -Q, np.inf, ask_depth)
    bid_depth = np.where(np.isfinite(bid_depth) & (bid_depth >= -3.0 / k), bid_depth, np.inf)
    ask_depth = np.where(np.isfinite(ask_depth) & (ask_depth >= -3.0 / k), ask_depth, np.inf)
    return bid_depth, ask_depth


def _compute_optimal_policy(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_q = 2 * Q + 1
    q_range = np.arange(-Q, Q + 1)
    q_float = q_range.astype(np.float64)

    # Fine grid for depths
    # Use log-spaced grid to capture small depths well
    # k * depth is the exponent. 
    # Optimal depth is typically small.
    # Let's use a grid in k*depth space.
    # min_kd = 0.01, max_kd = 10.0
    n_depths = 800
    kd_grid = np.geomspace(0.01, 10.0, n_depths)
    depth_grid = kd_grid / k
    p_fill = 1.0 - np.exp(-A * np.exp(-kd_grid) * dt)
    depth_p = depth_grid * p_fill

    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    inv_cost_factor = (gamma / 2.0) * sigma**2 * dt
    cost_vec = inv_cost_factor * q_float**2

    # Pre-compute terms for vectorization
    # p: (N,)
    # dp: (N,)
    # d: (N,)
    
    # We will loop backwards in time
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 40:
            # Fallback to heuristic-like values if running out of time
            # Just fill with current best or heuristic
            for n2 in range(n, -1, -1):
                for qi in range(n_q):
                    q = q_range[qi]
                    d = 1.0 / k + gamma * sigma**2 * q * (T - n2 * dt) / (A * k)
                    d = max(d, 0.01 / k)
                    bid_table[n2, qi] = d
                    ask_table[n2, qi] = d
            bid_table[:, n_q - 1] = np.inf
            ask_table[:, 0] = np.inf
            return bid_table, ask_table

        Vn1 = V[n + 1]
        
        # D1 = V(q+1) - V(q)
        # Dm = V(q-1) - V(q)
        V_plus = np.empty(n_q)
        V_minus = np.empty(n_q)
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1] # V(Q+1) is not defined, but for q=Q, we don't use D1
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0] # V(-Q-1) is not defined, but for q=-Q, we don't use Dm

        D1 = V_plus - Vn1
        Dm = V_minus - Vn1

        # Vectorized computation of J for all q and all (i, j)
        # J[q, i, j] = depth_p[i] + depth_p[j] + p[i]*D1[q]*(1-p[j]) + p[j]*Dm[q]*(1-p[i]) - cost[q]
        
        # Shapes:
        # p: (N,)
        # depth_p: (N,)
        # D1: (Q,)
        # Dm: (Q,)
        # cost: (Q,)
        
        # We need to broadcast to (Q, N, N)
        # p[i] -> (1, N, 1)
        # p[j] -> (1, 1, N)
        # D1[q] -> (Q, 1, 1)
        # Dm[q] -> (Q, 1, 1)
        # cost[q] -> (Q, 1, 1)
        
        p_b = p_fill[None, :, None]
        p_a = p_fill[None, None, :]
        dp_b = depth_p[None, :, None]
        dp_a = depth_p[None, None, :]
        
        D1_b = D1[:, None, None]
        Dm_b = Dm[:, None, None]
        cost_b = cost_vec[:, None, None]
        
        # Calculate J
        # Note: (1-p_a) and (1-p_b)
        term1 = dp_b + dp_a
        term2 = p_b * D1_b * (1.0 - p_a)
        term3 = p_a * Dm_b * (1.0 - p_b)
        
        J = term1 + term2 + term3 - cost_b
        
        # Find max indices
        # We want argmax over axes 1 and 2
        flat_indices = np.argmax(J, axis=(1, 2))
        
        # Map flat indices back to (i, j)
        # N = n_depths
        i_indices = flat_indices // n_depths
        j_indices = flat_indices % n_depths
        
        # Store optimal depths for interior states (1 to n_q-2)
        # q=0 and q=n_q-1 are handled separately
        
        bid_table[n, 1:n_q-1] = depth_grid[i_indices[1:n_q-1]]
        ask_table[n, 1:n_q-1] = depth_grid[j_indices[1:n_q-1]]
        
        # Update V for interior states
        # V[n, q] = J[q, i_max, j_max]
        # We can extract these values
        # J is (Q, N, N). We need J[q, i_q, j_q]
        # Using advanced indexing
        V[n, 1:n_q-1] = J[1:n_q-1, i_indices[1:n_q-1], j_indices[1:n_q-1]]
        
        # Handle boundaries
        # q = n_q - 1 (Inventory = Q)
        # No bid fill. p_b = 0.
        # J = d_a * p_a + p_a * Dm - cost
        # This is 1D optimization over j (ask depth)
        # J_boundary_q = depth_p[j] + p_fill[j] * Dm[n_q-1] - cost[n_q-1]
        # We can compute this for all j and find max
        # But Dm and cost are scalars for this q
        # We already have depth_p and p_fill vectors
        # J_q = depth_p + p_fill * Dm[n_q-1] - cost[n_q-1]
        # This is a vector of size N
        # Find argmax
        J_q_max = depth_p + p_fill * Dm[n_q - 1] - cost_vec[n_q - 1]
        j_max_q = np.argmax(J_q_max)
        ask_table[n, n_q - 1] = depth_grid[j_max_q]
        V[n, n_q - 1] = J_q_max[j_max_q]
        bid_table[n, n_q - 1] = np.inf # No bid
        
        # q = 0 (Inventory = -Q)
        # No ask fill. p_a = 0.
        # J = d_b * p_b + p_b * D1 - cost
        # J_boundary_q = depth_p[i] + p_fill[i] * D1[0] - cost[0]
        J_q_min = depth_p + p_fill * D1[0] - cost_vec[0]
        i_max_q = np.argmax(J_q_min)
        bid_table[n, 0] = depth_grid[i_max_q]
        V[n, 0] = J_q_min[i_max_q]
        ask_table[n, 0] = np.inf # No ask

    bid_table[:, n_q - 1] = np.inf
    ask_table[:, 0] = np.inf

    return bid_table, ask_table


def quote(t, inventory, mid, params):
    if time.time() - _START_TIME > TIME_LIMIT:
        return _heuristic_depths(params, t, inventory)

    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key not in _policy_cache:
        bid_table, ask_table = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_table, ask_table)
    else:
        bid_table, ask_table = _policy_cache[key]

    steps = params['steps']
    Q = params['Q']
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    q_index = inventory + Q
    q_index = np.clip(q_index, 0, 2 * Q)

    bid_depths = bid_table[n, q_index].copy()
    ask_depths = ask_table[n, q_index].copy()

    min_valid = -3.0 / params['k']
    bid_depths = np.where((bid_depths > min_valid) & np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where((ask_depths > min_valid) & np.isfinite(ask_depths), ask_depths, np.inf)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
