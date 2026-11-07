"""Optimal market making with inventory risk — precomputed DP policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_DEADLINE = 320.0  # safety margin before 330s kill
_start_time = time.time()


def _logsumexp(x, axis=None, b=None):
    """Numerically stable log-sum-exp along axis."""
    if b is not None:
        x = x * b
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0)
    s = np.sum(np.exp(x - m), axis=axis, keepdims=True)
    out = np.log(s) + m.squeeze(axis)
    return out


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """Solve the finite-horizon DP for the optimal bid/ask depths per inventory state."""
    dt = T / steps
    n_states = 2 * Q + 1
    # q = -Q, ..., 0, ..., Q -> index 0..n_states-1
    # inventory value at index i: q = i - Q
    q_vals = np.arange(n_states) - Q  # shape (n_states,)

    # Log-spaced depth grid: fine at small depths, up to 12/k
    n_grid = 80
    depths = np.concatenate([
        np.array([0.0]),
        np.geomspace(0.005, 12.0 / k, n_grid - 1)
    ])
    # Ensure minimum valid depth > -3/k (i.e., > 0 here since k>0)
    depths = depths[depths > 1e-9]
    depths = np.clip(depths, 1e-6, 12.0 / k)

    n_d = len(depths)
    # Precompute fill probabilities for all depths
    # p_fill(d) = 1 - exp(-A * exp(-k*d) * dt)
    exp_kd = np.exp(-k * depths)  # shape (n_d,)
    p_fill = 1.0 - np.exp(-A * exp_kd * dt)  # shape (n_d,)

    # Inventory cost per step: (gamma/2) sigma^2 q^2 dt
    inv_cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt  # shape (n_states,)

    # Value function: V[s, i] = optimal expected reward from step s onward at state i
    # We do backward induction: V[steps] = 0
    # V[s, i] = max_d_bid, d_ask [ reward + E[V[s+1, i']] ]
    #
    # To be efficient, we vectorize over states. For each state i, we choose
    # (d_bid, d_ask). The expected next-state value is:
    #   (1-pb)(1-pa) V[s+1, i] + pb(1-pa) V[s+1, i-1] + (1-pb)pa V[s+1, i+1] + pb*pa V[s+1, i]
    # with boundary conditions (no bid fill at q=Q, no ask fill at q=-Q).
    #
    # Reward = d_bid + d_ask - inv_cost[i]
    #
    # We'll iterate over depth choices. Since n_d=80, the full 2D grid is 80x80=6400
    # per state, times 61 states = ~390k evaluations per step. With 1000 steps that's
    # 390M — too slow in Python. Instead, we use the fact that the optimal bid and ask
    # depths are approximately separable. We'll do a 1D search: for each state, the
    # optimal bid depth and ask depth can be found by considering the marginal value.
    #
    # Better approach: since the mid is a martingale and the problem is symmetric
    # (up to inventory), we can use a policy iteration / value iteration approach.
    # But let's try a simpler approach: for each step, for each state, do a 1D
    # maximization over a combined "spread" parameter, or use the fact that
    # bid and ask choices are nearly independent.
    #
    # Actually, let's just do the full 2D but in a vectorized way over states.
    # For each (d_bid, d_ask) pair, compute the value for ALL states at once.

    V_next = np.zeros(n_states)  # V[steps] = 0

    # We'll store the optimal policy
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))

    # Precompute state transition indices
    # State i: q = i - Q
    # Bid fill: q decreases by 1 -> i-1 (if i > 0, i.e., q > -Q)
    # Ask fill: q increases by 1 -> i+1 (if i < n_states-1, i.e., q < Q)

    # For vectorization over states:
    # V_next[i-1] for bid fill (with boundary: if i=0, no bid fill)
    # V_next[i+1] for ask fill (with boundary: if i=n_states-1, no ask fill)

    V_next_minus = np.zeros(n_states)  # V_next[i-1], with V_next[-1] = V_next[0] (no fill)
    V_next_plus = np.zeros(n_states)   # V_next[i+1], with V_next[n_states] = V_next[n_states-1] (no fill)
    V_next_minus[1:] = V_next[:-1]
    V_next_minus[0] = V_next[0]  # no bid fill at q=-Q
    V_next_plus[:-1] = V_next[1:]
    V_next_plus[-1] = V_next[-1]  # no ask fill at q=Q

    # For each step (backward), for each depth pair, compute value for all states
    # We'll iterate over steps, and for each step, vectorize over depth pairs.

    # To keep it fast, we'll use a reduced grid for the 2D search.
    # Key insight: the optimal depths are typically in [0.01, 3/k].
    # Use a smaller 2D grid: 40x40 = 1600 pairs.

    n_grid_small = 40
    depths_small = np.geomspace(0.005, 12.0 / k, n_grid_small)
    depths_small = np.clip(depths_small, 1e-6, 12.0 / k)
    n_ds = len(depths_small)

    p_fill_small = 1.0 - np.exp(-A * np.exp(-k * depths_small) * dt)  # shape (n_ds,)

    # Precompute all (pb, pa) pairs
    pb_grid = p_fill_small[:, np.newaxis]  # shape (n_ds, 1)
    pa_grid = p_fill_small[np.newaxis, :]  # shape (1, n_ds)

    for s in range(steps - 1, -1, -1):
        # Update boundary-adjusted V_next values
        V_next_minus = np.zeros(n_states)
        V_next_plus = np.zeros(n_states)
        V_next_minus[1:] = V_next[:-1]
        V_next_minus[0] = V_next[0]
        V_next_plus[:-1] = V_next[1:]
        V_next_plus[-1] = V_next[-1]

        # For each depth pair (d_b, d_a), compute value for all states:
        # V[s, i] = d_b + d_a - inv_cost[i] + (1-pb)(1-pa)*V_next[i]
        #           + pb(1-pa)*V_next_minus[i] + (1-pb)*pa*V_next_plus[i]
        #           + pb*pa*V_next[i]
        #
        # = d_b + d_a - inv_cost[i] + (1-pb)(1-pa)*V_next[i]
        #   + pb(1-pa)*V_next_minus[i] + (1-pb)*pa*V_next_plus[i] + pb*pa*V_next[i]

        # Vectorize: for each (pb, pa) pair, compute the value array over states.
        # Shape: (n_ds, n_ds, n_states)
        # This is 40*40*61 = 97,600 floats — manageable.

        # Coefficients for V_next, V_next_minus, V_next_plus
        # coeff_V = (1-pb)(1-pa) + pb*pa
        # coeff_minus = pb*(1-pa)
        # coeff_plus = (1-pb)*pa
        pb_2d = p_fill_small[:, np.newaxis]  # (n_ds, 1)
        pa_2d = p_fill_small[np.newaxis, :]  # (1, n_ds)

        coeff_V = (1 - pb_2d) * (1 - pa_2d) + pb_2d * pa_2d  # (n_ds, n_ds)
        coeff_minus = pb_2d * (1 - pa_2d)  # (n_ds, n_ds)
        coeff_plus = (1 - pb_2d) * pa_2d  # (n_ds, n_ds)

        # Value for all states and all depth pairs:
        # val[db, da, i] = depths_small[db] + depths_small[da] - inv_cost[i]
        #   + coeff_V[db,da]*V_next[i] + coeff_minus[db,da]*V_next_minus[i] + coeff_plus[db,da]*V_next_plus[i]

        # Build as: base[db,da] + coeff_V[db,da,:,newaxis]*V_next[newaxis,newaxis,:]
        # Actually, let's do it step by step.

        # inv_cost is (n_states,), depths sum is (n_ds, n_ds)
        depth_sum = depths_small[:, np.newaxis] + depths_small[np.newaxis, :]  # (n_ds, n_ds)

        # val = depth_sum[:, :, np.newaxis] - inv_cost[np.newaxis, np.newaxis, :]
        #     + coeff_V[:, :, np.newaxis] * V_next[np.newaxis, np.newaxis, :]
        #     + coeff_minus[:, :, np.newaxis] * V_next_minus[np.newaxis, np.newaxis, :]
        #     + coeff_plus[:, :, np.newaxis] * V_next_plus[np.newaxis, np.newaxis, :]

        val = (depth_sum[:, :, np.newaxis]
               - inv_cost[np.newaxis, np.newaxis, :]
               + coeff_V[:, :, np.newaxis] * V_next[np.newaxis, np.newaxis, :]
               + coeff_minus[:, :, np.newaxis] * V_next_minus[np.newaxis, np.newaxis, :]
               + coeff_plus[:, :, np.newaxis] * V_next_plus[np.newaxis, np.newaxis, :])

        # Shape: (n_ds, n_ds, n_states)
        # Maximize over depth pairs for each state:
        best_val, best_idx = np.max(val, axis=(0, 1))  # shape (n_states,)
        # best_idx is the flat index into (n_ds, n_ds)
        best_db, best_da = np.unravel_index(best_idx, (n_ds, n_ds))

        V_next = best_val
        opt_bid[s] = depths_small[best_db]
        opt_ask[s] = depths_small[best_da]

        # Check deadline
        if time.time() > _DEADLINE:
            # Fall back to remaining steps with current best guess
            for s2 in range(s - 1, -1, -1):
                opt_bid[s2] = opt_bid[s]
                opt_ask[s2] = opt_ask[s]
            break

    return opt_bid, opt_ask, depths_small, n_ds


def _get_policy(params):
    """Get or compute the optimal policy for given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_optimal_policy(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """Return (bid_depths, ask_depths) arrays of shape (4000,)."""
    n = len(inventory)
    Q = params['Q']
    steps = params['steps']
    dt = params['T'] / steps
    s = int(round(t / dt))
    s = np.clip(s, 0, steps - 1)

    try:
        opt_bid, opt_ask, depths_small, n_ds = _get_policy(params)
        # inventory: shape (n,), values in [-Q, Q]
        # Map to state index: i = q + Q
        idx = np.clip(inventory + Q, 0, 2 * Q)
        bid_depths = opt_bid[s, idx]
        ask_depths = opt_ask[s, idx]
        # Ensure valid: > -3/k, no NaN, no -inf
        k = params['k']
        min_valid = -3.0 / k
        bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_valid), bid_depths, 1.0 / k)
        ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_valid), ask_depths, 1.0 / k)
        return bid_depths.astype(np.float64), ask_depths.astype(np.float64)
    except Exception:
        # Fallback: simple heuristic
        k = params['k']
        depth = 1.0 / k
        return np.full(n, depth), np.full(n, depth)


# EVOLVE-BLOCK-END
