"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

sys.setrecursionlimit(100000)

_start_time = time.time()
_DEADLINE = 300.0  # safety margin before 330s kill

# ---------------------------------------------------------------------------
# Cached exact (vectorized) dynamic-programming policy, keyed by regime.
# ---------------------------------------------------------------------------
_dp_cache = {}
_dp_cache_lock = None  # single-threaded; no lock needed


def _compute_dp_policy(sigma, A, k, gamma, T, steps, Q):
    """Solve the discrete optimal-control DP for this regime and return
    arrays bid_opt[q_idx], ask_opt[q_idx] for q in [-Q, Q] (shape (2Q+1,)).
    Uses a fine depth grid and vectorized Bellman updates.
    """
    dt = T / steps
    cost_c = 0.5 * gamma * sigma * sigma * dt

    # ---- depth grid (shared by bid and ask) ----
    dmin = -3.0 / k + 1e-6
    dmax = 5.0 / k
    N = 150
    dgrid = np.linspace(dmin, dmax, N)
    expkd = np.exp(-k * dgrid)
    pgrid = 1.0 - np.exp(-A * expkd * dt)   # fill prob at each depth

    # precompute profit = d * p  and "cost" = p (coefficient of value diff)
    profit = dgrid * pgrid
    # For a side with value-difference coef C (can be + or -), the objective
    # over depth is: profit + p * C  (for bid) or profit - p * C (for ask)
    # We'll just recompute the max over the small grid each step (cheap).

    # ---- value-difference arrays over q ----
    # q_idx: 0 -> q = -Q, ..., 2Q -> q = +Q
    # Vdiff[q_idx] = V(q) - V(q-1)   (for q > -Q)
    # Wdiff[q_idx] = V(q) - V(q+1)   (for q < +Q)
    # We store full-length arrays; entries outside the valid range are unused.
    NQ = 2 * Q + 1
    Vdiff = np.zeros(NQ)   # V(q) - V(q-1)
    Wdiff = np.zeros(NQ)   # V(q) - V(q+1)

    bid_opt = np.full(NQ, np.inf)
    ask_opt = np.full(NQ, np.inf)

    # helper: given a coefficient array C (per q), compute optimal depth
    # for the "bid" side: max_d [ d*p + p*C ] = max_d p*(d + C) + ... actually
    # objective = profit + p*C  where profit = d*p. So = p*d + p*C = p*(d+C).
    # For "ask": objective = profit - p*C = p*(d - C).
    def best_depth_bid(C):
        # C shape (NQ,); objective over grid: profit + pgrid*C
        # = (dgrid + C) * pgrid
        obj = (dgrid[None, :] + C[:, None]) * pgrid[None, :]
        idx = np.argmax(obj, axis=1)
        return dgrid[idx]

    def best_depth_ask(C):
        obj = (dgrid[None, :] - C[:, None]) * pgrid[None, :]
        idx = np.argmax(obj, axis=1)
        return dgrid[idx]

    # iterate backwards from t = steps-1 down to 0
    # At t, current q. After fills, q' in {q-1, q, q+1}.
    # E[V_{t+1}(q')] = p_b*p_a*V(q) + p_b*(1-p_a)*V(q-1) + (1-p_b)*p_a*V(q+1) + (1-p_b)(1-p_a)V(q)
    #   = V(q) + p_b*(1-p_a)*(V(q-1)-V(q)) + (1-p_b)*p_a*(V(q+1)-V(q))
    #   = V(q) - p_b*(1-p_a)*Vdiff[q] + (1-p_b)*p_a*Wdiff[q]
    # where Vdiff[q]=V(q)-V(q-1), Wdiff[q]=V(q)-V(q+1).
    #
    # Bellman for V_t(q):
    #   V_t(q) = -cost_c*q^2 + max_{db,da} [ db*pb + da*pa + V_{t+1}(q)
    #             - pb*(1-pa)*Vdiff[q] + (1-pb)*pa*Wdiff[q] ]
    #   = -cost_c*q^2 + V_{t+1}(q)
    #     + max_db [ db*pb - pb*(1-pa)*Vdiff[q] ]   (separable in db)
    #     + max_da [ da*pa + (1-pb)*pa*Wdiff[q] ]
    # For the bid max: factor pb:  max_db pb*(db - (1-pa)*Vdiff[q])
    #   -> choose db to maximize (db - Cb)*pb where Cb=(1-pa)*Vdiff[q].
    #   Note (1-pa) depends on da, so not strictly separable! But the
    #   (1-pa) factor is in [0,1] and multiplies Vdiff. To keep it tractable
    #   and accurate, we treat the coupling by iterating: the dominant term is
    #   db*pb; the correction -(1-pa)*Vdiff*pb is small. We approximate by
    #   using the *expected* (1-pa) ~ (1 - p_a_star) where p_a_star is the
    #   fill prob at the optimal ask depth. We do one fixed-point refinement.

    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    q_sq = q_vals * q_vals

    for step in range(steps - 1, -1, -1):
        # ---- ask side (depends on Wdiff[q] and (1-pb)) ----
        # We first compute optimal ask using (1-pb) approximated by its
        # value at the symmetric base, then refine. Use (1-pb) ~ (1-pb0)
        # where pb0 is fill prob at the base depth 1/k.
        pb0 = 1.0 - np.exp(-A * np.exp(-k * (1.0 / k)) * dt)
        one_minus_pb0 = 1.0 - pb0
        # ask objective: pa*(da + (1-pb)*Wdiff[q])  -> maximize pa*(da + one_minus_pb*Wdiff)
        C_ask = one_minus_pb0 * Wdiff
        ask_d = best_depth_ask(C_ask)
        pa_at = pgrid[np.searchsorted(dgrid, ask_d)]
        one_minus_pb = 1.0 - pa_at  # refine: (1-pb) where pb at optimal bid? use pa as proxy

        # ---- bid side ----
        # bid objective: pb*(db - (1-pa)*Vdiff[q])  -> maximize pb*(db - one_minus_pa*Vdiff)
        C_bid = one_minus_pb * Vdiff
        bid_d = best_depth_bid(C_bid)
        pb_at = pgrid[np.searchsorted(dgrid, bid_d)]
        one_minus_pa = 1.0 - pb_at

        # one refinement pass for the coupling
        C_ask2 = one_minus_pa * Wdiff
        ask_d = best_depth_ask(C_ask2)
        pa_at = pgrid[np.searchsorted(dgrid, ask_d)]
        one_minus_pb = 1.0 - pa_at
        C_bid2 = one_minus_pb * Vdiff
        bid_d = best_depth_bid(C_bid2)
        pb_at = pgrid[np.searchsorted(dgrid, bid_d)]
        one_minus_pa = 1.0 - pb_at

        # ---- compute new value differences V_t(q)-V_t(q-1) and V_t(q)-V_t(q+1) ----
        # V_t(q) = -cost_c*q^2 + V_{t+1}(q) + bid_term(q) + ask_term(q)
        # bid_term(q) = pb*(db - one_minus_pa*Vdiff[q]) at optimal (db, one_minus_pa)
        # ask_term(q) = pa*(da + one_minus_pa... wait ask uses (1-pb)*Wdiff)
        #   ask_term(q) = pa*(da + one_minus_pb*Wdiff[q])
        bid_term = pb_at * (bid_d - one_minus_pa * Vdiff)
        ask_term = pa_at * (ask_d + one_minus_pb * Wdiff)
        Vt = -cost_c * q_sq + bid_term + ask_term   # (V_{t+1}(q) cancels in differences)

        # Vdiff_new[q] = Vt(q) - Vt(q-1); valid for q_idx >= 1 (q > -Q)
        Vdiff_new = np.zeros(NQ)
        Vdiff_new[1:] = Vt[1:] - Vt[:-1]
        # Wdiff_new[q] = Vt(q) - Vt(q+1); valid for q_idx <= NQ-2 (q < Q)
        Wdiff_new = np.zeros(NQ)
        Wdiff_new[:-1] = Vt[:-1] - Vt[1:]

        # store optimal depths for THIS step's q (we only need the final step's
        # policy? No -- the policy depends on t. But the evaluator calls quote
        # at each t with the SAME params; the optimal depth at step t depends on
        # how many steps remain. So we must store the policy per step.)
        # However memory for steps x NQ is fine (1000 x 61).
        bid_opt[step] = None  # placeholder; we store in 2D below
        # We'll collect into 2D arrays.

        Vdiff = Vdiff_new
        Wdiff = Wdiff_new

    return None  # not used; real storage handled below


def _build_policy(sigma, A, k, gamma, T, steps, Q):
    """Return (bid_table, ask_table) each of shape (steps, 2Q+1) where
    table[step, q_idx] is the optimal depth (or np.inf at boundaries)."""
    dt = T / steps
    cost_c = 0.5 * gamma * sigma * sigma * dt

    dmin = -3.0 / k + 1e-6
    dmax = 5.0 / k
    N = 150
    dgrid = np.linspace(dmin, dmax, N)
    pgrid = 1.0 - np.exp(-A * np.exp(-k * dgrid) * dt)

    NQ = 2 * Q + 1
    bid_table = np.full((steps, NQ), np.inf)
    ask_table = np.full((steps, NQ), np.inf)

    Vdiff = np.zeros(NQ)
    Wdiff = np.zeros(NQ)

    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    q_sq = q_vals * q_vals

    def best_depth(C, sign):
        # sign=+1 -> maximize (d + C)*p ; sign=-1 -> maximize (d - C)*p
        obj = (dgrid[None, :] + sign * C[:, None]) * pgrid[None, :]
        idx = np.argmax(obj, axis=1)
        return dgrid[idx]

    for step in range(steps - 1, -1, -1):
        one_minus_pb0 = 1.0 - (1.0 - np.exp(-A * np.exp(-k * (1.0 / k)) * dt))
        C_ask = one_minus_pb0 * Wdiff
        ask_d = best_depth(C_ask, +1)
        pa_at = pgrid[np.searchsorted(dgrid, ask_d)]
        one_minus_pb = 1.0 - pa_at
        C_bid = one_minus_pb * Vdiff
        bid_d = best_depth(C_bid, -1)
        pb_at = pgrid[np.searchsorted(dgrid, bid_d)]
        one_minus_pa = 1.0 - pb_at
        # refine once
        C_ask = one_minus_pa * Wdiff
        ask_d = best_depth(C_ask, +1)
        pa_at = pgrid[np.searchsorted(dgrid, ask_d)]
        one_minus_pb = 1.0 - pa_at
        C_bid = one_minus_pb * Vdiff
        bid_d = best_depth(C_bid, -1)
        pb_at = pgrid[np.searchsorted(dgrid, bid_d)]
        one_minus_pa = 1.0 - pb_at

        bid_term = pb_at * (bid_d - one_minus_pa * Vdiff)
        ask_term = pa_at * (ask_d + one_minus_pb * Wdiff)
        Vt = -cost_c * q_sq + bid_term + ask_term

        Vdiff_new = np.zeros(NQ)
        Vdiff_new[1:] = Vt[1:] - Vt[:-1]
        Wdiff_new = np.zeros(NQ)
        Wdiff_new[:-1] = Vt[:-1] - Vt[1:]
        Vdiff, Wdiff = Vdiff_new, Wdiff_new

        bid_table[step] = bid_d
        ask_table[step] = ask_d

    # boundary: no bid fill at q = Q, no ask fill at q = -Q
    bid_table[:, Q + Q] = np.inf       # q_idx for q = +Q
    ask_table[:, 0] = np.inf           # q_idx for q = -Q
    return bid_table, ask_table


def _get_policy(params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"],
           params["T"], params["steps"], params["Q"])
    if key in _dp_cache:
        return _dp_cache[key]
    if time.time() - _start_time > _DEADLINE - 30.0:
        # not enough time; fall back to analytic (no caching)
        return None
    bt, at = _build_policy(key[0], key[1], key[2], key[3], key[4], key[5], key[6])
    _dp_cache[key] = (bt, at)
    return (bt, at)


def _analytic_depths(q, params):
    """Fast closed-form fallback (current HWHM-style)."""
    sigma = params["sigma"]; A = params["A"]; k = params["k"]
    gamma = params["gamma"]; T = params["T"]; steps = params["steps"]; Q = params["Q"]
    dt = T / steps
    n_rem = max(1, steps - int(round(t_global / dt)))
    d_0 = 1.0 / k
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


t_global = 0.0


def quote(t, inventory, mid, params):
    """Optimal market making quote via cached vectorized DP policy."""
    global t_global
    t_global = t

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
        return _analytic_depths(q, params)

    bid_table, ask_table = pol
    q_idx = (q + Q).astype(np.int64)
    q_idx = np.clip(q_idx, 0, 2 * Q)
    d_bid = bid_table[step, q_idx]
    d_ask = ask_table[step, q_idx]

    # enforce boundaries strictly
    d_bid = np.where(q >= Q, np.inf, d_bid)
    d_ask = np.where(q <= -Q, np.inf, d_ask)

    d_bid = np.asarray(d_bid, dtype=np.float64)
    d_ask = np.asarray(d_ask, dtype=np.float64)
    return (d_bid, d_ask)

# EVOLVE-BLOCK-END
