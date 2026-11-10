"""Rewritten program: precompute HJB value function, then quote via argmax."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

sys.setrecursionlimit(100000)

_start_time = time.time()
_DEADLINE = 325.0

# Cache for precomputed value functions per regime
_value_cache = {}
# Cache for precomputed optimal depth arrays
_opt_depth_cache = {}


def _precompute_value(params):
    """
    Precompute the optimal value function V[q, n] and optimal depths
    for all inventory levels q in [-Q, Q] and all steps n in [0, steps-1].
    
    Uses dynamic programming (backward induction) on the discrete model.
    The state is (q, n) where n is the step index.
    The action is (d_bid, d_ask) where d_bid >= 0, d_ask >= 0.
    
    For each (q, n), we find the optimal depths by considering the expected
    value of taking each possible depth combination.
    
    Since the fill probability depends on depth via 1 - exp(-A exp(-k d) dt),
    and the reward from a fill is the depth itself, we can optimize over depth.
    
    For a given q and n, the value is:
    V(q, n) = max_{d_b, d_a} {
        P_b * d_b + P_a * d_a - (gamma/2) * sigma^2 * q^2 * dt
        + E[V(q', n+1)]
    }
    where:
      P_b = (1 - exp(-A exp(-k d_b) dt)) if q < Q else 0
      P_a = (1 - exp(-A exp(-k d_a) dt)) if q > -Q else 0
      q' = q + 1 (if ask fill) - 1 (if bid fill), with the constraint that
            no bid fill at q=Q, no ask fill at q=-Q
    
    The mid moves by sigma*sqrt(dt)*N(0,1), but since the reward and future
    value depend on q and n (not mid directly, because the objective only has
    the inventory cost term and fill rewards which are depth-based), the mid
    movement doesn't affect the optimal depths. Wait, let me re-read.
    
    Actually, the objective is: depths earned minus sum of (gamma/2)*sigma^2*q^2*dt.
    The mid movement affects where the quotes are placed relative to the market,
    but since the fill probability only depends on the depth (distance from mid),
    and the reward is the depth itself, the mid movement doesn't enter the
    optimization directly. The inventory q is the only state variable that
    matters for the value function.
    
    So V depends only on q and n.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    n_levels = 2 * Q + 1
    # q ranges from -Q to Q, index = q + Q
    
    # V[q_idx, n] = optimal value from step n with inventory q
    V = np.zeros((n_levels, steps + 1), dtype=np.float64)
    # Optimal depths
    opt_db = np.full((n_levels, steps), np.inf, dtype=np.float64)
    opt_da = np.full((n_levels, steps), np.inf, dtype=np.float64)
    
    # Terminal: V[q, steps] = 0 for all q
    
    # We need to search over possible depths. The fill probability is
    # P(d) = 1 - exp(-A * exp(-k*d) * dt)
    # The reward from a fill is d.
    # So the expected reward from bid side is: P(d_b) * d_b
    # We want to maximize P(d_b)*d_b + (future value terms)
    
    # For each depth d, the contribution is P(d)*d where P(d) = 1 - exp(-A*exp(-k*d)*dt)
    # Let's precompute a grid of depths and their P(d)*d values.
    
    # The optimal depth for a single fill (ignoring future) maximizes P(d)*d
    # P(d) = 1 - exp(-lambda) where lambda = A*exp(-k*d)*dt
    # P(d)*d = d * (1 - exp(-A*exp(-k*d)*dt))
    
    # Let's use a fine grid of depths.
    # The relevant range: d from about 0 to maybe 5/k or so.
    # For very large d, P(d) -> 0, so P(d)*d -> 0.
    # For d = 0, P(0) = 1 - exp(-A*dt), P(0)*0 = 0.
    
    # Let's sample depths on a log-spaced or linear grid.
    d_max = 5.0 / k
    n_d = 200
    d_grid = np.linspace(0, d_max, n_d)
    
    # P(d) for each d in grid
    lam = A * np.exp(-k * d_grid) * dt
    P = 1.0 - np.exp(-lam)
    reward = P * d_grid  # P(d) * d
    
    # For each q and n, we need to choose d_b and d_a independently (since
    # bid and ask fills are independent), so we can optimize each separately.
    # But the future value depends on q' which depends on both fills.
    # So we need to consider the joint distribution.
    
    # The possible outcomes:
    # - No fill: prob (1-P_b)(1-P_a), q' = q
    # - Bid fill only: prob P_b(1-P_a), q' = q-1 (if q>0)
    # - Ask fill only: prob (1-P_b)P_a, q' = q+1 (if q<Q)
    # - Both: prob P_b*P_a, q' = q (bid and ask cancel)
    
    # Wait, if both fill, inventory doesn't change? Let me re-read.
    # "A fill earns its depth and moves inventory one unit"
    # "no bid fill at q = Q, no ask fill at q = -Q"
    # So if both fill, bid moves q by -1, ask moves q by +1, net 0.
    
    # So q' = q - (bid_fill ? 1 : 0) + (ask_fill ? 1 : 0)
    
    # The expected future value is:
    # (1-P_b)(1-P_a) * V(q, n+1)
    # + P_b(1-P_a) * V(q-1, n+1)  [if q-1 >= -Q, else this term is 0 since no bid fill at q=Q... wait]
    # Actually, if q = Q, no bid fill, so P_b = 0.
    # If q = -Q, no ask fill, so P_a = 0.
    
    # Let me handle boundaries:
    # If q >= Q: P_b = 0 (no bid fill)
    # If q <= -Q: P_a = 0 (no ask fill)
    
    # For the DP, I'll iterate n from steps-1 down to 0.
    
    # Precompute for each d_grid entry: P, and the reward P*d
    # Also precompute the "cost" of inventory: (gamma/2)*sigma^2*q^2*dt
    
    inv_cost = (gamma / 2.0) * sigma * sigma * dt
    q_vals = np.arange(-Q, Q + 1)
    inv_cost_arr = inv_cost * q_vals * q_vals
    
    for n in range(steps - 1, -1, -1):
        for qi in range(n_levels):
            q = qi - Q  # actual inventory
            cost = inv_cost_arr[qi]
            
            if q >= Q:
                # No bid fill possible
                P_b_arr = np.zeros(n_d)
                # q' if bid fill: not possible
                V_qm1 = 0.0  # unused
            else:
                P_b_arr = P.copy()
                V_qm1 = V[qi - 1, n + 1] if qi - 1 >= 0 else 0.0
            
            if q <= -Q:
                # No ask fill possible
                P_a_arr = np.zeros(n_d)
                V_qp1 = 0.0  # unused
            else:
                P_a_arr = P.copy()
                V_qp1 = V[qi + 1, n + 1] if qi + 1 < n_levels else 0.0
            
            V_q = V[qi, n + 1]
            
            best_val = -np.inf
            best_db = np.inf
            best_da = np.inf
            
            # For each combination of d_b and d_a from the grid
            # This is O(n_d^2) per (q, n), which is 200^2 = 40000 per state
            # Total states: 61 * 1000 = 61000
            # Total ops: 61000 * 40000 = 2.44e9 -- too slow!
            
            # Need to be smarter. Since bid and ask are independent in their
            # fill probabilities, and the future value is linear in the
            # probabilities, we can separate the optimization.
            
            # Expected future value:
            # (1-P_b)(1-P_a) * V_q
            # + P_b(1-P_a) * V_qm1  [if q < Q]
            # + (1-P_b)P_a * V_qp1  [if q > -Q]
            # + P_b*P_a * V_q  [both fill, q unchanged]
            
            # = V_q * [(1-P_b)(1-P_a) + P_b*P_a]
            #   + (1-P_a) * P_b * V_qm1
            #   + (1-P_b) * P_a * V_qp1
            
            # = V_q * [1 - P_b - P_a + 2*P_b*P_a]
            #   + P_b * V_qm1 - P_b*P_a * V_qm1
            #   + P_a * V_qp1 - P_b*P_a * V_qp1
            
            # = V_q + P_b*(V_qm1 - V_q) + P_a*(V_qp1 - V_q)
            #   + P_b*P_a*(2*V_q - V_qm1 - V_qp1)
            
            # So the total objective for given P_b, P_a:
            # P_b * d_b + P_a * d_a - cost
            # + V_q + P_b*(V_qm1 - V_q) + P_a*(V_qp1 - V_q)
            #   + P_b*P_a*(2*V_q - V_qm1 - V_qp1)
            
            # Let a = V_qm1 - V_q, b = V_qp1 - V_q, c = 2*V_q - V_qm1 - V_qp1
            # = V_q - cost + P_b*(d_b + a) + P_a*(d_a + b) + P_b*P_a*c
            
            # Since P_b depends only on d_b and P_a depends only on d_a,
            # we can't fully separate, but we can do:
            # For each d_b (giving P_b), optimize over d_a.
            # Or use the fact that the cross term P_b*P_a*c is small.
            
            # Let's just do a 1D search over d_b and for each, a 1D search over d_a.
            # But that's still O(n_d^2).
            
            # Alternative: since n_d=200, n_d^2=40000, and 61000 states,
            # that's 2.44e9 operations. In numpy, vectorized, this might be
            # feasible if we batch.
            
            # Let me vectorize over d_b and d_a simultaneously.
            P_b_2d = P_b_arr[:, None]  # (n_d, 1)
            P_a_2d = P_a_arr[None, :]  # (1, n_d)
            d_b_2d = d_grid[:, None]
            d_a_2d = d_grid[None, :]
            
            a_val = (V_qm1 - V_q) if q < Q else 0.0
            b_val = (V_qp1 - V_q) if q > -Q else 0.0
            c_val = (2.0 * V_q - V_qm1 - V_qp1) if (q < Q and q > -Q) else 0.0
            
            # Objective matrix
            obj = (V_q - cost
                   + P_b_2d * (d_b_2d + a_val)
                   + P_a_2d * (d_a_2d + b_val)
                   + P_b_2d * P_a_2d * c_val)
            
            idx = np.unravel_index(np.argmax(obj), obj.shape)
            best_db = d_grid[idx[0]]
            best_da = d_grid[idx[1]]
            best_val = obj[idx]
            
            opt_db[qi, n] = best_db
            opt_da[qi, n] = best_da
            V[qi, n] = best_val
    
    return V, opt_db, opt_da


def quote(t, inventory, mid, params):
    """
    Optimal market making quote via precomputed HJB value function.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        k = params["k"]
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    
    # Use a cache key based on regime parameters
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    
    if cache_key not in _opt_depth_cache:
        # Precompute (this is the expensive part)
        V, opt_db, opt_da = _precompute_value(params)
        _opt_depth_cache[cache_key] = (opt_db, opt_da)
    
    opt_db, opt_da = _opt_depth_cache[cache_key]
    
    q = np.asarray(inventory, dtype=np.int32)
    n_levels = 2 * Q + 1
    
    # Index into the optimal depth arrays
    q_idx = q + Q
    q_idx = np.clip(q_idx, 0, n_levels - 1)
    
    d_bid = opt_db[q_idx, n].astype(np.float64)
    d_ask = opt_da[q_idx, n].astype(np.float64)
    
    # Boundary conditions: no bid at q=Q, no ask at q=-Q
    d_bid = np.where(q >= Q, np.inf, d_bid)
    d_ask = np.where(q <= -Q, np.inf, d_ask)
    
    # Ensure valid depths
    min_depth = -3.0 / k + 1e-9
    max_depth = 50.0 / k
    
    d_bid = np.where(np.isfinite(d_bid), np.clip(d_bid, min_depth, max_depth), d_bid)
    d_ask = np.where(np.isfinite(d_ask), np.clip(d_ask, min_depth, max_depth), d_ask)
    
    d_bid = np.asarray(d_bid, dtype=np.float64)
    d_ask = np.asarray(d_ask, dtype=np.float64)
    
    return (d_bid, d_ask)

# EVOLVE-BLOCK-END
