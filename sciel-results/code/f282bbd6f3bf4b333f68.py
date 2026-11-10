"""Optimal market making with inventory risk - HJB continuous approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

sys.setrecursionlimit(100000)

_start_time = time.time()
_DEADLINE = 280.0  # safety margin before 330s kill

# ---------------------------------------------------------------------------
# Cached policy, keyed by regime.
# ---------------------------------------------------------------------------
_policy_cache = {}


def _solve_hjb_policy(sigma, A, k, gamma, T, steps, Q):
    """Solve the continuous HJB via semi-Lagrangian on a uniform inventory grid.
    
    The value function V(t, q) satisfies the HJB:
      -V_t = inf_q [cost + ...]  (minimization form)
    or equivalently for our maximization objective:
      V_t + cost(q) + min over depths of [expected cost of fills] = 0
    
    We use the analytical first-order condition for optimal depth to eliminate
    the inner optimization, giving a closed-form depth in terms of the value
    function derivatives.
    
    Returns (bid_table, ask_table) of shape (steps, 2Q+1).
    """
    dt = T / steps
    cost_c = 0.5 * gamma * sigma * sigma * dt
    NQ = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    q_sq = q_vals * q_vals
    
    # Precompute the fill probability function and its derivative
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    # p'(d) = A * k * exp(-k*d) * exp(-A*exp(-k*d)*dt) * dt
    # For the FOC: maximize p(d) * (d + C) where C is the continuation value term
    # d/dd [p(d)*(d+C)] = p'(d)*(d+C) + p(d) = 0
    # => p'(d)/p(d) = -1/(d+C)
    # => A*k*exp(-k*d)*exp(-A*exp(-k*d)*dt)*dt / (1-exp(-A*exp(-k*d)*dt)) = -1/(d+C)
    
    # This is transcendental, so we solve it numerically per (q, step)
    # But we can precompute on a fine grid and interpolate
    
    # For speed, use a moderate grid
    dmin = -3.0 / k + 1e-6
    dmax = 10.0 / k
    N = 400
    dgrid = np.linspace(dmin, dmax, N)
    
    # p(d) and p'(d) on the grid
    exp_kd = np.exp(-k * dgrid)
    lambda_d = A * exp_kd * dt
    pgrid = 1.0 - np.exp(-lambda_d)
    # p'(d) = A * k * exp(-k*d) * exp(-lambda_d) * dt
    pprime_grid = A * k * exp_kd * np.exp(-lambda_d) * dt
    
    # For the FOC: p'(d)*(d+C) + p(d) = 0
    # => p'(d)/p(d) = -1/(d+C)
    # Let r(d) = p'(d)/p(d) = A*k*exp(-k*d)*dt / (exp(lambda_d) - 1) ... 
    # Actually: p'(d)/p(d) = A*k*exp(-k*d)*exp(-lambda_d)*dt / (1 - exp(-lambda_d))
    #           = A*k*exp(-k*d)*dt / (exp(lambda_d) - 1)
    # For large lambda_d: p'(d)/p(d) ≈ A*k*exp(-k*d)*dt/lambda_d = k*exp(-k*d)
    # For small lambda_d: p'(d)/p(d) ≈ A*k*exp(-k*d)*dt/(lambda_d) = k*exp(-k*d)
    # So approximately p'(d)/p(d) ≈ k*exp(-k*d) for all d (when lambda_d is small or large)
    # More precisely: p'(d)/p(d) = k * exp(-k*d) * lambda_d / (exp(lambda_d) - 1)
    
    lambda_d_arr = A * exp_kd * dt
    # Handle lambda_d = 0 case
    ratio = np.where(lambda_d_arr > 1e-12, 
                     lambda_d_arr / (np.exp(lambda_d_arr) - 1.0),
                     1.0)
    # p'(d)/p(d) = k * exp(-k*d) * ratio
    r_grid = k * exp_kd * ratio
    
    # FOC: r(d) = -1/(d+C) => r(d)*(d+C) = -1
    # => d*r(d) + C*r(d) = -1
    # => C = -1/r(d) - d
    # So for a given C, find d such that C = -1/r(d) - d
    # i.e., f(d) = -1/r(d) - d - C = 0
    
    # Precompute -1/r(d) - d for the grid
    # r(d) can be very small for large d, so -1/r(d) can be very negative
    # Let's compute the "effective C" for each grid point
    # C_eff(d) = -1/r(d) - d
    # When r(d) is small, -1/r(d) is large negative
    # When r(d) is large, -1/r(d) is small negative
    
    # Actually, let's think about this differently.
    # The objective is max_d p(d)*(d+C)
    # For the bid: C = -Vdiff (negative of value difference)
    # For the ask: C = +Wdiff (value difference)
    
    # Let's just do a vectorized argmax on a fine grid per step.
    # With N=400 and NQ=61, that's 400*61 = 24400 operations per step, * 1000 steps = 24.4M
    # That should be fast enough.
    
    bid_table = np.full((steps, NQ), np.inf)
    ask_table = np.full((steps, NQ), np.inf)
    
    # Value differences: Vdiff[i] = V(q_i) - V(q_{i-1}), Wdiff[i] = V(q_i) - V(q_{i+1})
    Vdiff = np.zeros(NQ)
    Wdiff = np.zeros(NQ)
    
    # Base fill probability estimate for initialization
    base_d = 1.0 / k
    base_lambda = A * np.exp(-k * base_d) * dt
    p_base = 1.0 - np.exp(-base_lambda)
    one_minus_pb = 1.0 - p_base
    one_minus_pa = 1.0 - p_base
    
    for step in range(steps - 1, -1, -1):
        # Ask: maximize p(d) * (d + C_ask) where C_ask = one_minus_pb * Wdiff
        C_ask = one_minus_pb * Wdiff
        # obj_ask(d) = pgrid * (dgrid + C_ask[:,None])
        obj_ask = pgrid[None, :] * (dgrid[None, :] + C_ask[:, None])
        idx_ask = np.argmax(obj_ask, axis=1)
        ask_d = dgrid[idx_ask]
        pa_at = pgrid[idx_ask]
        
        # Bid: maximize p(d) * (d - C_bid) where C_bid = one_minus_pa * Vdiff
        # i.e., maximize p(d) * (d + C_bid_neg) where C_bid_neg = -one_minus_pa * Vdiff
        C_bid_neg = -one_minus_pa * Vdiff
        obj_bid = pgrid[None, :] * (dgrid[None, :] + C_bid_neg[:, None])
        idx_bid = np.argmax(obj_bid, axis=1)
        bid_d = dgrid[idx_bid]
        pb_at = pgrid[idx_bid]
        
        # One refinement pass
        one_minus_pa = 1.0 - pb_at
        one_minus_pb = 1.0 - pa_at
        
        C_ask = one_minus_pb * Wdiff
        obj_ask = pgrid[None, :] * (dgrid[None, :] + C_ask[:, None])
        idx_ask = np.argmax(obj_ask, axis=1)
        ask_d = dgrid[idx_ask]
        pa_at = pgrid[idx_ask]
        
        C_bid_neg = -one_minus_pa * Vdiff
        obj_bid = pgrid[None, :] * (dgrid[None, :] + C_bid_neg[:, None])
        idx_bid = np.argmax(obj_bid, axis=1)
        bid_d = dgrid[idx_bid]
        pb_at = pgrid[idx_bid]
        
        # Bellman update
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
    
    # Boundaries
    bid_table[:, 2 * Q] = np.inf
    ask_table[:, 0] = np.inf
    
    return bid_table, ask_table


def _get_policy(params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"],
           params["T"], params["steps"], params["Q"])
    if key in _policy_cache:
        return _policy_cache[key]
    if time.time() - _start_time > _DEADLINE - 20.0:
        return None
    bt, at = _solve_hjb_policy(key[0], key[1], key[2], key[3], key[4], key[5], key[6])
    _policy_cache[key] = (bt, at)
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
