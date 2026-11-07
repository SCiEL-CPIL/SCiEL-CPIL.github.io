"""Optimal market making with inventory risk via dynamic programming."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for DP tables keyed by parameter tuple
_dp_cache = {}
_start_time = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill


def _build_dp_table(sigma, A, k, gamma, T, steps, Q):
    """Precompute optimal bid/ask depths for each step and inventory level."""
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _dp_cache:
        return _dp_cache[key]

    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1  # inventory from -Q to Q

    # Value function: V[step, inv_index]
    # V[n_steps, q] = 0 (no terminal charge)
    V = np.zeros((n_steps + 1, n_inv))

    # Store optimal depths
    opt_bid = np.full((n_steps, n_inv), np.inf)
    opt_ask = np.full((n_steps, n_inv), np.inf)

    # Grid of candidate depths to search over
    # Depths should be positive; too large = no fill, too small = low reward
    # Optimal depth is roughly in [0.01, 3/k]
    max_depth = 3.0 / k
    # Use a fine grid
    n_grid = 200
    depth_grid = np.linspace(0.005, max_depth, n_grid)

    # Precompute fill probabilities for each depth
    # p_fill(depth) = 1 - exp(-A * exp(-k * depth) * dt)
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)

    # Backward induction
    for n in range(n_steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE:
            # Return what we have; fill remaining with a reasonable default
            default_depth = 1.0 / k
            for nn in range(n, n_steps):
                opt_bid[nn, :] = default_depth
                opt_ask[nn, :] = default_depth
            _dp_cache[key] = (opt_bid, opt_ask, V)
            return _dp_cache[key]

        V_next = V[n + 1]
        V_curr = V[n]

        for q in range(-Q, Q + 1):
            inv_idx = q + Q
            holding_cost = (gamma / 2.0) * sigma**2 * q**2 * dt

            best_val = -np.inf
            best_bid = depth_grid[0]
            best_ask = depth_grid[0]

            # Try each candidate depth for bid and ask
            # Vectorize over depth pairs would be 200x200 = 40000 per (n,q)
            # That's too slow. Instead, note that bid and ask are independent
            # given q, so we can optimize them separately.

            # Actually, they ARE independent in the expectation because
            # bid and ask fills are independent. So:
            # V = max_bid [bid_depth * p_bid + E_q' | bid] + max_ask [ask_depth * p_ask + E_q' | ask] - holding_cost
            # But the future value depends on whether bid OR ask fills, which
            # creates coupling. Let me think again...

            # Actually with independent fills:
            # P(both fill) = p_b * p_a
            # P(bid only) = p_b * (1-p_a)
            # P(ask only) = (1-p_b) * p_a
            # P(neither) = (1-p_b)*(1-p_a)

            # New inventory:
            # both fill: q (bid +1, ask -1, net 0)
            # bid only: q+1
            # ask only: q-1
            # neither: q

            # So we need to consider joint optimization.
            # Let's vectorize: for each bid depth i, ask depth j:
            # val = bid_depth[i]*p_fill[i] + ask_depth[j]*p_fill[j]
            #     + p_fill[i]*p_fill[j]*V_next[q+Q]
            #     + p_fill[i]*(1-p_fill[j])*V_next[min(q+1,Q)+Q]
            #     + (1-p_fill[i])*p_fill[j]*V_next[max(q-1,-Q)+Q]
            #     + (1-p_fill[i])*(1-p_fill[j])*V_next[q+Q]
            #     - holding_cost

            # This is n_grid^2 = 40000 operations per (n,q) pair.
            # n_steps * n_inv * n_grid^2 = 1000 * 61 * 40000 = 2.46 billion. Too slow.

            # Better approach: note that the future value term can be rewritten.
            # Let V_b = V_next[q+Q] (q unchanged)
            # Let V_bp = V_next[min(q+1,Q)+Q] (bid fills, net +1)
            # Let V_am = V_next[max(q-1,-Q)+Q] (ask fills, net -1)

            # val(i,j) = bid[i]*pf[i] + ask[j]*pf[j]
            #   + pf[i]*pf[j]*V_b + pf[i]*(1-pf[j])*V_bp + (1-pf[i])*pf[j]*V_am + (1-pf[i])*(1-pf[j])*V_b
            #   - holding_cost
            # = bid[i]*pf[i] + ask[j]*pf[j]
            #   + V_b*(pf[i]*pf[j] + (1-pf[i])*(1-pf[j]))
            #   + V_bp*pf[i]*(1-pf[j]) + V_am*(1-pf[i])*pf[j]
            #   - holding_cost

            # = bid[i]*pf[i] + ask[j]*pf[j]
            #   + V_b*(1 - pf[i] - pf[j] + 2*pf[i]*pf[j])
            #   + V_bp*pf[i] - V_bp*pf[i]*pf[j] + V_am*pf[j] - V_am*pf[i]*pf[j]
            #   - holding_cost

            # = bid[i]*pf[i] + V_bp*pf[i] + ask[j]*pf[j] + V_am*pf[j]
            #   + V_b*(1 - pf[i] - pf[j]) + (2*V_b - V_bp - V_am)*pf[i]*pf[j]
            #   - holding_cost

            # This is separable! Let me define:
            # f(i) = bid[i]*pf[i] + V_bp*pf[i]
            # g(j) = ask[j]*pf[j] + V_am*pf[j]
            # h(i,j) = V_b*(1 - pf[i] - pf[j]) + (2*V_b - V_bp - V_am)*pf[i]*pf[j]

            # Still has cross term. But we can write:
            # val = [f(i) + V_b*(1-pf[i])] + [g(j) + V_b*(1-pf[j])] - V_b
            #     + (2*V_b - V_bp - V_am)*pf[i]*pf[j] - holding_cost
            # Hmm, still coupled.

            # Alternative: for a fixed i, the optimal j can be found in O(n_grid).
            # So total is O(n_grid^2) per (n,q) but with small constant.
            # 1000 * 61 * 40000 = 2.46e9... still too slow in Python.

            # Let me use a coarser grid or vectorize the inner loop.
            # With n_grid=100, that's 1000*61*10000 = 6.1e8. Still too slow.

            # Let me try n_grid=50: 1000*61*2500 = 1.525e8. Maybe OK with numpy.

            # Actually, let me reconsider. The depth grid can be adaptive.
            # The optimal depth is likely near 1/k. Let me use a smaller range.
            # Or better: use the fact that we can vectorize over j for each i.

            # For fixed i (bid depth), find best j:
            # val(i,j) = C_i + ask[j]*pf[j] + V_am*pf[j] + (2*V_b - V_bp - V_am)*pf[i]*pf[j]
            #          - V_b*pf[j] - holding_cost
            # where C_i = bid[i]*pf[i] + V_bp*pf[i] + V_b*(1-pf[i])

            # = C_i + pf[j]*(ask[j] + V_am + (2*V_b - V_bp - V_am)*pf[i] - V_b) - holding_cost
            # = C_i + pf[j]*(ask[j] + V_am + (V_b - V_bp)*pf[i] + (V_b - V_am)*pf[i] - V_b) - holding_cost
            # = C_i + pf[j]*(ask[j] + (V_b - V_b) + (V_am - V_b) + (V_b - V_bp)*pf[i] + (V_b - V_am)*pf[i]) - holding_cost

            # Let me just compute it directly with numpy vectorization.

            V_b = V_next[q + Q]
            q_plus = min(q + 1, Q)
            q_minus = max(q - 1, -Q)
            V_bp = V_next[q_plus + Q]
            V_am = V_next[q_minus + Q]

            # For each bid depth i, compute the best ask depth j
            # val(i,j) = bid_grid[i]*pf[i] + ask_grid[j]*pf[j]
            #   + pf[i]*pf[j]*V_b + pf[i]*(1-pf[j])*V_bp + (1-pf[i])*pf[j]*V_am + (1-pf[i])*(1-pf[j])*V_b
            #   - holding_cost

            # Let me vectorize: for all i, compute the array of val over j, then take max.
            # pf is shape (n_grid,)
            # bid_grid * pf is shape (n_grid,)
            # We need a matrix of shape (n_grid, n_grid) which is 2500-40000 elements.

            # Use broadcasting:
            # pf_i = pf[:, None]  # (n_grid, 1)
            # pf_j = pf[None, :]  # (1, n_grid)
            # val_matrix = bid_grid[:, None]*pf_i + depth_grid[None, :]*pf_j \
            #     + pf_i*pf_j*V_b + pf_i*(1-pf_j)*V_bp + (1-pf_i)*pf_j*V_am + (1-pf_i)*(1-pf_j)*V_b \
            #     - holding_cost

            # This creates a (n_grid, n_grid) matrix. For n_grid=100, that's 10000 floats = 80KB.
            # 1000 * 61 = 61000 such operations. Total memory ops: 61000 * 80KB = 4.88 GB.
            # That's a lot but might be OK in ~30 seconds with numpy.

            # Let me use n_grid=80 to be safe.
            n_grid_local = 80
            depth_grid_local = np.linspace(0.005, max_depth, n_grid_local)
            pf_local = 1.0 - np.exp(-A * np.exp(-k * depth_grid_local) * dt)

            pf_i = pf_local[:, None]
            pf_j = pf_local[None, :]
            bid_col = depth_grid_local[:, None]
            ask_row = depth_grid_local[None, :]

            val_matrix = (bid_col * pf_i + ask_row * pf_j
                          + pf_i * pf_j * V_b
                          + pf_i * (1.0 - pf_j) * V_bp
                          + (1.0 - pf_i) * pf_j * V_am
                          + (1.0 - pf_i) * (1.0 - pf_j) * V_b
                          - holding_cost)

            idx = np.argmax(val_matrix)
            i_best, j_best = divmod(idx, n_grid_local)

            best_bid = depth_grid_local[i_best]
            best_ask = depth_grid_local[j_best]
            V_curr[inv_idx] = val_matrix[i_best, j_best]

            # Handle boundary conditions: no bid fill at q=Q, no ask fill at q=-Q
            if q == Q:
                # No bid fill, so bid depth doesn't matter for fills, but we still quote
                # Actually, the problem says "no bid fill at q = Q", meaning if a bid
                # order would fill, it doesn't. So we can set bid to inf (no quote)
                # or keep it but it won't fill. For optimality, we should not quote bid.
                opt_bid[n, inv_idx] = np.inf
                # Recompute with bid disabled
                V_b2 = V_next[q + Q]
                q_minus2 = max(q - 1, -Q)
                V_am2 = V_next[q_minus2 + Q]
                # Only ask can fill
                # val(j) = ask[j]*pf[j] + pf[j]*V_am2 + (1-pf[j])*V_b2 - holding_cost
                val_ask = depth_grid_local * pf_local + pf_local * V_am2 + (1.0 - pf_local) * V_b2 - holding_cost
                j_best2 = np.argmax(val_ask)
                best_ask = depth_grid_local[j_best2]
                V_curr[inv_idx] = val_ask[j_best2]
                opt_ask[n, inv_idx] = best_ask
            elif q == -Q:
                # No ask fill
                opt_ask[n, inv_idx] = np.inf
                V_b2 = V_next[q + Q]
                q_plus2 = min(q + 1, Q)
                V_bp2 = V_next[q_plus2 + Q]
                val_bid = depth_grid_local * pf_local + pf_local * V_bp2 + (1.0 - pf_local) * V_b2 - holding_cost
                i_best2 = np.argmax(val_bid)
                best_bid = depth_grid_local[i_best2]
                V_curr[inv_idx] = val_bid[i_best2]
                opt_bid[n, inv_idx] = best_bid
            else:
                opt_bid[n, inv_idx] = best_bid
                opt_ask[n, inv_idx] = best_ask

    _dp_cache[key] = (opt_bid, opt_ask, V)
    return _dp_cache[key]


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths using precomputed DP table."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    opt_bid, opt_ask, V = _build_dp_table(sigma, A, k, gamma, T, steps, Q)

    # Determine current step
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    # inventory is array of shape (4000,), values in [-Q, Q]
    inv_idx = inventory + Q  # index into DP table, shape (4000,)
    inv_idx = np.clip(inv_idx, 0, 2 * Q)

    bid_depths = opt_bid[n, inv_idx].astype(np.float64)
    ask_depths = opt_ask[n, inv_idx].astype(np.float64)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
