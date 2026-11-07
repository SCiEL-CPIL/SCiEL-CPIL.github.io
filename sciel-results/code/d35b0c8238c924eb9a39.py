"""Optimal market making with inventory risk - DP precomputation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precomputed optimal policy cache
_policy_cache = {}


def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    """Solve the DP problem and return optimal bid/ask depth tables."""
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1  # inventory from -Q to Q

    # Depth grid: log-spaced for finer resolution at small depths
    n_depths = 500
    d_min = 0.001
    d_max = 5.0 / k
    depth_grid = np.geomspace(d_min, d_max, n_depths)
    # Include 0 (no quote equivalent, very small depth ~ 0)
    depth_grid = np.concatenate([[0.0], depth_grid])
    n_d = len(depth_grid)

    # Fill probabilities for each depth
    # p_fill(d) = 1 - exp(-A * exp(-k * d) * dt)
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)

    # Value function: V[t, inv] where inv in 0..n_inv-1 corresponds to q = -Q + inv
    # V[n_steps, q] = 0 for all q (no terminal charge)
    V = np.zeros((n_steps + 1, n_inv))

    # Optimal depth indices
    opt_bid = np.zeros((n_steps, n_inv), dtype=np.int32)
    opt_ask = np.zeros((n_steps, n_inv), dtype=np.int32)

    # Backward induction
    for t in range(n_steps - 1, -1, -1):
        V_next = V[t + 1]
        # Inventory cost for this step
        q_vals = np.arange(-Q, Q + 1)  # shape (n_inv,)
        cost = (gamma / 2.0) * sigma**2 * q_vals**2 * dt  # shape (n_inv,)

        V_t = np.zeros(n_inv)
        for i in range(n_inv):
            q = -Q + i
            # Boundary conditions
            can_bid = (q < Q)  # can have bid fill
            can_ask = (q > -Q)  # can have ask fill

            if not can_bid and not can_ask:
                # At boundaries where no fills possible
                V_t[i] = -cost[i] + V_next[i]
                opt_bid[t, i] = 0
                opt_ask[t, i] = 0
                continue

            # Compute expected next value for each (bid, ask) combination
            if can_bid:
                pb = p_fill  # shape (n_d,)
                db = depth_grid  # shape (n_d,)
                V_bid = V_next[i + 1]
            else:
                pb = np.array([0.0])
                db = np.array([0.0])
                V_bid = 0.0

            if can_ask:
                pa = p_fill  # shape (n_d,)
                da = depth_grid  # shape (n_d,)
                V_ask = V_next[i - 1]
            else:
                pa = np.array([0.0])
                da = np.array([0.0])
                V_ask = 0.0

            n_pb = len(pb)
            n_pa = len(pa)

            # Vectorize: create matrices
            pb_mat = pb[:, np.newaxis]  # (n_pb, 1)
            pa_mat = pa[np.newaxis, :]  # (1, n_pa)
            db_mat = db[:, np.newaxis]  # (n_pb, 1)
            da_mat = da[np.newaxis, :]  # (1, n_pa)

            E_depth = pb_mat * db_mat - pa_mat * da_mat  # (n_pb, n_pa)

            E_Vnext = ((1 - pb_mat) * (1 - pa_mat) * V_next[i]
                       + pb_mat * (1 - pa_mat) * V_bid
                       + (1 - pb_mat) * pa_mat * V_ask
                       + pb_mat * pa_mat * V_next[i])  # (n_pb, n_pa)

            total = E_depth + E_Vnext - cost[i]  # (n_pb, n_pa)

            # Find optimal (bid_idx, ask_idx)
            idx = np.unravel_index(np.argmax(total), total.shape)
            opt_bid[t, i] = idx[0]
            opt_ask[t, i] = idx[1]
            V_t[i] = total[idx]

        V[t] = V_t

    # Convert optimal indices to depths
    bid_depths_table = depth_grid[opt_bid]  # shape (n_steps, n_inv)
    ask_depths_table = depth_grid[opt_ask]  # shape (n_steps, n_inv)

    return bid_depths_table, ask_depths_table, depth_grid


def _get_policy(params):
    """Get or compute the optimal policy for given params."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _policy_cache:
        _policy_cache[key] = _solve_dp(sigma, A, k, gamma, T, steps, Q)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths."""
    t_start = time.time()
    deadline = t_start + 290  # Safety margin before 300s limit

    try:
        bid_table, ask_table, depth_grid = _get_policy(params)

        steps = params['steps']
        Q = params['Q']

        # Determine current time step
        dt = params['T'] / steps
        t_step = int(round(t / dt))
        t_step = max(0, min(t_step, steps - 1))

        # Map inventory to index: inv_idx = q + Q
        inv_idx = inventory + Q
        inv_idx = np.clip(inv_idx, 0, 2 * Q)

        # Look up optimal depths
        bid_depths = bid_table[t_step, inv_idx]
        ask_depths = ask_table[t_step, inv_idx]

        # Ensure valid: depths must be >= 0 and finite
        bid_depths = np.maximum(bid_depths, 0.0)
        ask_depths = np.maximum(ask_depths, 0.0)
        bid_depths = np.nan_to_num(bid_depths, nan=1.0/params['k'], posinf=10.0/params['k'])
        ask_depths = np.nan_to_num(ask_depths, nan=1.0/params['k'], posinf=10.0/params['k'])

        return bid_depths, ask_depths
    except Exception:
        # Fallback: risk-neutral depth
        k = params['k']
        depth = 1.0 / k
        n = len(inventory)
        return np.full(n, depth), np.full(n, depth)
# EVOLVE-BLOCK-END
