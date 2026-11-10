"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

sys.setrecursionlimit(100000)

_start_time = time.time()
_DEADLINE = 300.0  # safety margin before 330s kill

# ---------------------------------------------------------------------------
# Cached DP policy, keyed by regime.
# ---------------------------------------------------------------------------
_dp_cache = {}


def _build_policy(sigma, A, k, gamma, T, steps, Q):
    """Return (bid_table, ask_table) each of shape (steps, 2Q+1) where
    table[step, q_idx] is the optimal depth (or np.inf at boundaries)."""
    dt = T / steps
    cost_c = 0.5 * gamma * sigma * sigma * dt

    # Depth grid: fine enough to capture the optimum accurately
    dmin = -3.0 / k + 1e-6
    dmax = 8.0 / k
    N = 200
    dgrid = np.linspace(dmin, dmax, N)
    pgrid = 1.0 - np.exp(-A * np.exp(-k * dgrid) * dt)

    NQ = 2 * Q + 1
    bid_table = np.full((steps, NQ), np.inf)
    ask_table = np.full((steps, NQ), np.inf)

    # Value differences: Vdiff[q_idx] = V(q) - V(q-1), Wdiff[q_idx] = V(q) - V(q+1)
    Vdiff = np.zeros(NQ)
    Wdiff = np.zeros(NQ)

    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    q_sq = q_vals * q_vals

    # Precompute for fast argmax: we maximize (dgrid + sign*C) * pgrid over dgrid
    def best_depth(C, sign):
        """C shape (NQ,); returns optimal depth for each q."""
        obj = (dgrid[None, :] + sign * C[:, None]) * pgrid[None, :]
        idx = np.argmax(obj, axis=1)
        return dgrid[idx]

    # Initial estimate for (1-pb) and (1-pa) using a base depth
    base_depth = 1.0 / k
    p_base = 1.0 - np.exp(-A * np.exp(-k * base_depth) * dt)
    one_minus_pb = 1.0 - p_base
    one_minus_pa = 1.0 - p_base

    for step in range(steps - 1, -1, -1):
        # --- Solve for optimal ask depth ---
        # Ask objective: maximize pa * (da + (1-pb) * Wdiff[q])
        # => maximize (dgrid + one_minus_pb * Wdiff) * pgrid
        C_ask = one_minus_pb * Wdiff
        ask_d = best_depth(C_ask, +1)
        pa_at = pgrid[np.searchsorted(dgrid, ask_d)]

        # --- Solve for optimal bid depth ---
        # Bid objective: maximize pb * (db - (1-pa) * Vdiff[q])
        # => maximize (dgrid - one_minus_pa * Vdiff) * pgrid
        C_bid = one_minus_pa * Vdiff
        bid_d = best_depth(C_bid, -1)
        pb_at = pgrid[np.searchsorted(dgrid, bid_d)]

        # --- One refinement pass for the coupling ---
        one_minus_pa = 1.0 - pb_at
        one_minus_pb = 1.0 - pa_at

        C_ask = one_minus_pb * Wdiff
        ask_d = best_depth(C_ask, +1)
        pa_at = pgrid[np.searchsorted(dgrid, ask_d)]

        C_bid = one_minus_pa * Vdiff
        bid_d = best_depth(C_bid, -1)
        pb_at = pgrid[np.searchsorted(dgrid, bid_d)]

        # --- Compute value increments for Bellman update ---
        # V_t(q) - V_{t+1}(q) = -cost_c*q^2 + bid_term + ask_term
        bid_term = pb_at * (bid_d - one_minus_pa * Vdiff)
        ask_term = pa_at * (ask_d + one_minus_pb * Wdiff)
        Vt = -cost_c * q_sq + bid_term + ask_term

        # Update value differences
        Vdiff_new = np.zeros(NQ)
        Vdiff_new[1:] = Vt[1:] - Vt[:-1]
        Wdiff_new = np.zeros(NQ)
        Wdiff_new[:-1] = Vt[:-1] - Vt[1:]
        Vdiff, Wdiff = Vdiff_new, Wdiff_new

        bid_table[step] = bid_d
        ask_table[step] = ask_d

    # Boundary conditions: no bid at q=Q, no ask at q=-Q
    bid_table[:, 2 * Q] = np.inf
    ask_table[:, 0] = np.inf
    return bid_table, ask_table


def _get_policy(params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"],
           params["T"], params["steps"], params["Q"])
    if key in _dp_cache:
        return _dp_cache[key]
    if time.time() - _start_time > _DEADLINE - 30.0:
        return None
    bt, at = _build_policy(key[0], key[1], key[2], key[3], key[4], key[5], key[6])
    _dp_cache[key] = (bt, at)
    return (bt, at)


def _analytic_depths(q, t, params):
    """Fast closed-form fallback using HWHM-style inventory skewing."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    n_rem = max(1, steps - int(round(t / dt)))
    d_0 = 1.0 / k
    # Correct skew: depth = d_0 - DeltaV where DeltaV is linear in q
    # DeltaV ≈ gamma * sigma^2 * dt * n_rem * q + 0.5 * gamma * sigma^2 * dt
    alpha = gamma * sigma * sigma * dt * n_rem
    offset = 0.5 * alpha
    d_bid = d_0 + alpha * q + offset
    d_ask = d_0 - alpha * q + offset
    min_depth = -3.0 / k + 1e-6
    max_depth = 50.0 / k
    d_bid = np.clip(d_bid, min_depth, max_depth)
    d_ask = np.clip(d_ask, min_depth, max_depth)
    d_bid = np.where(q >= Q, np.inf, d_bid)
    d_ask = np.where(q <= -Q, np.inf, d_ask)
    return np.asarray(d_bid, dtype=np.float64), np.asarray(d_ask, dtype=np.float64)


def quote(t, inventory, mid, params):
    """Optimal market making quote via cached DP policy."""
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        k = params["k"]
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()

    Q = params["Q"]
    q = np.asarray(inventory, dtype=np.float64)
    steps = params["steps"]
    dt = params["T"] / steps
    step = min(steps - 1, max(0, int(round(t / dt))))

    pol = _get_policy(params)
    if pol is None:
        return _analytic_depths(q, t, params)

    bid_table, ask_table = pol
    q_idx = (q + Q).astype(np.int64)
    q_idx = np.clip(q_idx, 0, 2 * Q)
    d_bid = bid_table[step, q_idx]
    d_ask = ask_table[step, q_idx]

    # Enforce boundaries strictly
    d_bid = np.where(q >= Q, np.inf, d_bid)
    d_ask = np.where(q <= -Q, np.inf, d_ask)

    d_bid = np.asarray(d_bid, dtype=np.float64)
    d_ask = np.asarray(d_ask, dtype=np.float64)
    return (d_bid, d_ask)

# EVOLVE-BLOCK-END
