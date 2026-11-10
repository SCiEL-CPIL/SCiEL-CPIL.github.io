"""Optimal market making with inventory risk - full 2D grid DP + local refinement.

Uses a fine 2D grid to find the joint (bid, ask) optimum at each state,
then applies local golden-section refinement for sub-grid accuracy.
"""
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_TIME_BUDGET = 320.0  # seconds, leave 10s margin before 330s kill


def _fill_prob_vec(depths, A, k, dt):
    """Vectorized fill probability."""
    depths = np.asarray(depths, dtype=float)
    p = np.where(depths > 0, 1.0 - np.exp(-A * np.exp(-k * depths) * dt), 0.0)
    return np.clip(p, 0.0, 1.0)


def _fill_prob_scalar(depth, A, k, dt):
    """Scalar fill probability."""
    if depth <= 0:
        return 0.0
    p = 1.0 - np.exp(-A * np.exp(-k * depth) * dt)
    return min(max(p, 0.0), 1.0)


def _golden_section_1d(f, a, b, tol=1e-7, max_iter=50):
    """Golden section search for maximum of f on [a, b]."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc > fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
    x_opt = (a + b) / 2.0
    return x_opt


def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes the optimal bid and ask depths using:
    1. Fine 2D grid for joint (bid, ask) search at each state
    2. Golden-section local refinement for sub-grid accuracy
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1

    # Inventory holding cost per step
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt

    # Grid parameters
    # Use a fine 2D grid. The depth range [0.005, 6.0] covers all relevant regimes.
    n_grid = 120
    grid = np.linspace(0.005, 6.0, n_grid)

    # V[n, q_idx]
    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))

    start_time = time.time()
    deadline = start_time + (_TIME_BUDGET - (time.time() - _GLOBAL_START))

    # Pre-compute fill probabilities on grid
    p_grid = _fill_prob_vec(grid, A, k, dt)

    # Build 2D arrays
    # p_b[i, j] = fill prob for bid depth grid[i]
    # p_a[i, j] = fill prob for ask depth grid[j]
    P_b = np.tile(p_grid[:, np.newaxis], (1, n_grid))  # shape (n_grid, n_grid), varies with row
    P_a = np.tile(p_grid[np.newaxis, :], (n_grid, 1))  # shape (n_grid, n_grid), varies with col

    Depths_b = np.tile(grid[:, np.newaxis], (1, n_grid))
    Depths_a = np.tile(grid[np.newaxis, :], (n_grid, 1))
    Rewards = Depths_b * P_b + Depths_a * P_a

    P_plus = P_b * (1.0 - P_a)
    P_minus = P_a * (1.0 - P_b)
    P_same = 1.0 - P_plus - P_minus

    # Number of golden-section refinement rounds
    n_refine = 3

    for n in range(steps - 1, -1, -1):
        # Time check
        if time.time() > deadline:
            if n < steps - 1:
                V[n:n+1, :] = V[n+1, :]
                opt_bid[n:n+1, :] = opt_bid[n+1, :]
                opt_ask[n:n+1, :] = opt_ask[n+1, :]
            break

        V_next = V[n + 1]

        # Transition inventory offsets
        V_q_plus = np.zeros(n_q)
        V_q_minus = np.zeros(n_q)
        V_q = V_next.copy()

        V_q_plus[:n_q - 1] = V_next[1:]
        V_q_minus[1:] = V_next[:-1]

        # Compute values on 2D grid for each q
        # Values[q_idx, i, j] = total value for bid=grid[i], ask=grid[j]
        Values = Rewards[np.newaxis, :, :] - C_inv[:, np.newaxis, np.newaxis]
        Values += P_plus[np.newaxis, :, :] * V_q_plus[:, np.newaxis, np.newaxis]
        Values += P_minus[np.newaxis, :, :] * V_q_minus[:, np.newaxis, np.newaxis]
        Values += P_same[np.newaxis, :, :] * V_q[:, np.newaxis, np.newaxis]

        # Boundary handling
        Values[-1, :, :] -= P_plus * V_q_plus[-1, np.newaxis, np.newaxis]
        Values[0, :, :] -= P_minus * V_q_minus[0, np.newaxis, np.newaxis]

        # Find best (i, j) on grid for each q
        flat_values = Values.reshape(n_q, -1)
        max_indices = np.argmax(flat_values, axis=1)
        i_idx, j_idx = np.unravel_index(max_indices, (n_grid, n_grid))

        refined_bid = np.zeros(n_q)
        refined_ask = np.zeros(n_q)
        V_refined = np.zeros(n_q)

        for q_idx in range(n_q):
            b_cur = grid[i_idx[q_idx]]
            a_cur = grid[j_idx[q_idx]]

            # Golden-section refinement: alternate optimizing bid and ask
            for ref_iter in range(n_refine):
                # Optimize bid given ask
                if q_idx == n_q - 1:
                    b_opt = b_cur
                else:
                    def obj_b(b):
                        pb = _fill_prob_scalar(b, A, k, dt)
                        pa = _fill_prob_scalar(a_cur, A, k, dt)
                        if q_idx == 0:
                            trans = pb * V_q_plus[q_idx] + (1 - pb) * V_q[q_idx]
                            return b * pb - C_inv[q_idx] + trans
                        else:
                            trans = pb * (1 - pa) * V_q_plus[q_idx] + \
                                    pa * (1 - pb) * V_q_minus[q_idx] + \
                                    (1 - pb - pa + pb * pa) * V_q[q_idx]
                            return b * pb + a_cur * pa - C_inv[q_idx] + trans

                    # Search range: expand around current
                    b_lo = max(0.001, b_cur - 0.5)
                    b_hi = min(8.0, b_cur + 0.5)
                    b_opt = _golden_section_1d(obj_b, b_lo, b_hi, tol=1e-8, max_iter=40)

                # Optimize ask given bid
                if q_idx == 0:
                    a_opt = a_cur
                else:
                    def obj_a(a):
                        pb = _fill_prob_scalar(b_opt, A, k, dt)
                        pa = _fill_prob_scalar(a, A, k, dt)
                        if q_idx == n_q - 1:
                            trans = pa * V_q_minus[q_idx] + (1 - pa) * V_q[q_idx]
                            return a * pa - C_inv[q_idx] + trans
                        else:
                            trans = pb * (1 - pa) * V_q_plus[q_idx] + \
                                    pa * (1 - pb) * V_q_minus[q_idx] + \
                                    (1 - pb - pa + pb * pa) * V_q[q_idx]
                            return b_opt * pb + a * pa - C_inv[q_idx] + trans

                    a_lo = max(0.001, a_cur - 0.5)
                    a_hi = min(8.0, a_cur + 0.5)
                    a_opt = _golden_section_1d(obj_a, a_lo, a_hi, tol=1e-8, max_iter=40)

                b_cur = b_opt
                a_cur = a_opt

            refined_bid[q_idx] = b_cur
            refined_ask[q_idx] = a_cur

            # Compute refined value
            pb = _fill_prob_scalar(b_cur, A, k, dt)
            pa = _fill_prob_scalar(a_cur, A, k, dt)

            if q_idx == n_q - 1:
                trans = pa * V_q_minus[q_idx] + (1 - pa) * V_q[q_idx]
                V_refined[q_idx] = a_cur * pa - C_inv[q_idx] + trans
            elif q_idx == 0:
                trans = pb * V_q_plus[q_idx] + (1 - pb) * V_q[q_idx]
                V_refined[q_idx] = b_cur * pb - C_inv[q_idx] + trans
            else:
                trans = pb * (1 - pa) * V_q_plus[q_idx] + \
                        pa * (1 - pb) * V_q_minus[q_idx] + \
                        (1 - pb - pa + pb * pa) * V_q[q_idx]
                V_refined[q_idx] = b_cur * pb + a_cur * pa - C_inv[q_idx] + trans

        V[n, :] = V_refined
        opt_bid[n, :] = refined_bid
        opt_ask[n, :] = refined_ask

    return opt_bid, opt_ask


def _get_policy(params):
    """Retrieves or computes the optimal policy for the given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'],
        params['T'], params['steps'], params['Q']
    )

    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """Returns optimal bid and ask depths based on pre-computed DP policy."""
    opt_bid, opt_ask = _get_policy(params)

    dt = params['T'] / params['steps']
    n = int(round(t / dt))

    if n < 0:
        n = 0
    if n >= params['steps']:
        n = params['steps'] - 1

    Q = params['Q']
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)

    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]

    # Ensure valid outputs
    bid_depths = np.where(np.isnan(bid_depths), 0.01, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), 0.01, ask_depths)
    bid_depths = np.where(bid_depths < 0, 0.01, bid_depths)
    ask_depths = np.where(ask_depths < 0, 0.01, ask_depths)

    return bid_depths.astype(float), ask_depths.astype(float)
