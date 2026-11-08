"""Optimal market making with continuous-depth FOC-based policy."""
import numpy as np
import time

_policy_cache = {}
_GLOBAL_START = None


def _compute_policy(params):
    """
    Compute optimal bid/ask depths via backward induction.
    At each (step, inventory) state, solve the FOC for the optimal depth
    that maximizes the Bellman expression.
    Returns (bid_depths, ask_depths) of shape (steps, 2*Q+1).
    """
    global _GLOBAL_START
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)

    # Value function
    V = np.zeros(n_states)

    # Optimal actions storage
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))

    # Cost per state
    cost = 0.5 * gamma * sigma**2 * (q_values**2) * dt

    # Boundary masks
    no_bid = np.zeros(n_states, dtype=bool)
    no_bid[-1] = True  # q = Q: no bid fill
    no_ask = np.zeros(n_states, dtype=bool)
    no_ask[0] = True   # q = -Q: no ask fill

    # Precompute: for the FOC, we need to solve for d in:
    # max_d { p_b(d)*(d + Delta_b) + p_a(d)*(d + Delta_a) - p_b(d)*p_a(d)*Curv + V - cost }
    # where p(d) = 1 - exp(-A*exp(-k*d)*dt)
    #
    # For bid depth d_b, the FOC w.r.t. d_b is:
    # dp_b/dd_b * (d_b + Delta_b) + p_b - dp_b/dd_b * p_a * Curv = 0
    # where dp_b/dd_b = A*k*dt*exp(-k*d_b)*exp(-A*exp(-k*d_b)*dt)
    #                 = k * (1 - p_b) * exp(-k*d_b) * A * dt  ... let me re-derive
    #
    # p(d) = 1 - exp(-A*exp(-k*d)*dt)
    # dp/dd = A*exp(-k*d)*dt * exp(-A*exp(-k*d)*dt) = A*exp(-k*d)*dt*(1-p(d))
    #
    # FOC for d_b: dp_b/dd_b * (d_b + Delta_b) + p_b - dp_b/dd_b * p_a * Curv = 0
    # => dp_b/dd_b * (d_b + Delta_b - p_a * Curv) + p_b = 0
    # => A*exp(-k*d_b)*dt*(1-p_b) * (d_b + Delta_b - p_a*Curv) + p_b = 0
    #
    # This is transcendental. We'll solve it numerically with Newton's method
    # but vectorized over states.

    # We'll use a hybrid: for each step, compute Delta_b, Delta_a, Curv,
    # then solve FOC for each state.

    def solve_foc(Delta, p_other, Curv, is_no_quote_mask):
        """
        Solve FOC for depth d given:
          dp/dd * (d + Delta - p_other * Curv) + p = 0
        where p = 1 - exp(-A*exp(-k*d)*dt)
        dp/dd = A*k*dt*exp(-k*d)*(1-p)  ... wait let me redo.
        
        Actually dp/dd = A * exp(-k*d) * dt * exp(-A*exp(-k*d)*dt)
                      = A * exp(-k*d) * dt * (1 - p)
        
        FOC: A*exp(-k*d)*dt*(1-p)*(d + Delta - p_other*Curv) + p = 0
        
        Let f(d) = A*exp(-k*d)*dt*(1-p(d))*(d + Delta - p_other*Curv) + p(d)
        We want f(d) = 0.
        
        For states where is_no_quote_mask is True, return 1/k (arbitrary, won't be used).
        """
        n = len(Delta)
        # Initialize: use 1/k as starting point
        d = np.full(n, 1.0 / k)
        
        # Only solve for non-masked states
        active = ~is_no_quote_mask
        if not np.any(active):
            return d
        
        d_active = d[active]
        Delta_a = Delta[active]
        p_other_a = p_other[active]
        Curv_a = Curv[active]
        
        # Newton's method, vectorized
        for _ in range(15):
            exp_kd = np.exp(-k * d_active)
            exp_term = np.exp(-A * exp_kd * dt)
            p = 1.0 - exp_term
            dp_dd = A * exp_kd * dt * exp_term  # = A*exp(-k*d)*dt*(1-p)
            
            # f(d) = dp_dd * (d + Delta - p_other*Curv) + p
            bracket = d_active + Delta_a - p_other_a * Curv_a
            f = dp_dd * bracket + p
            
            # f'(d): derivative of f w.r.t. d
            # d(dp_dd)/dd = -k * dp_dd  (since dp_dd = A*exp(-k*d)*dt*(1-p), 
            #   d/dd = -k*A*exp(-k*d)*dt*(1-p) + A*exp(-k*d)*dt*(-dp/dd)
            #   = -k*dp_dd - A*exp(-k*d)*dt*dp_dd = -dp_dd*(k + A*exp(-k*d)*dt))
            # Actually let me just use: dp_dd = A*exp(-k*d)*dt*(1-p)
            # d(dp_dd)/dd = -k*A*exp(-k*d)*dt*(1-p) + A*exp(-k*d)*dt*(-dp/dd)
            #             = -k*dp_dd - A*exp(-k*d)*dt*dp_dd
            #             = -dp_dd*(k + A*exp(-k*d)*dt)
            
            ddp_dd_dd = -dp_dd * (k + A * exp_kd * dt)
            
            # f'(d) = ddp_dd_dd * bracket + dp_dd * 1 + dp_dd
            #        = ddp_dd_dd * bracket + 2*dp_dd
            fp = ddp_dd_dd * bracket + 2.0 * dp_dd
            
            # Newton step
            step = f / np.where(np.abs(fp) > 1e-15, fp, 1e-15)
            d_active = d_active - step
            
            # Clamp to valid range: d > 0 and d < 30/k (avoid extreme)
            d_active = np.clip(d_active, 1e-6, 30.0 / k)
        
        d[active] = d_active
        return d

    # Backward induction
    for n in range(steps - 1, -1, -1):
        elapsed = time.time() - _GLOBAL_START
        if elapsed > 280.0:
            # Fallback
            for nn in range(n, -1, -1):
                opt_bid[nn, :] = 1.0 / k
                opt_ask[nn, :] = 1.0 / k
            break

        # Compute state differences
        Delta_b = np.zeros(n_states)  # V[q-1] - V[q]  (bid fill decreases q)
        Delta_a = np.zeros(n_states)  # V[q+1] - V[q]  (ask fill increases q)
        Curv = np.zeros(n_states)     # V[q-1] + V[q+1] - 2*V[q]

        if n_states > 2:
            Delta_b[1:-1] = V[:-2] - V[1:-1]
            Delta_a[1:-1] = V[2:] - V[1:-1]
            Curv[1:-1] = V[:-2] + V[2:] - 2.0 * V[1:-1]
        Delta_b[0] = V[1] - V[0]
        Delta_a[-1] = V[-2] - V[-1]

        # Solve FOC for bid depth
        # For bid: p_other is the ask probability at the optimal ask depth
        # But we have a coupled system! bid FOC depends on p_a, ask FOC depends on p_b.
        # Solution: iterate. Start with p_b = p_a = 0, then alternate.
        
        # Initialize
        d_bid = np.full(n_states, 1.0 / k)
        d_ask = np.full(n_states, 1.0 / k)
        
        # Iterate to convergence
        for _iter in range(10):
            # Compute p at current depths
            exp_kd_b = np.exp(-k * d_bid)
            p_b = 1.0 - np.exp(-A * exp_kd_b * dt)
            exp_kd_a = np.exp(-k * d_ask)
            p_a = 1.0 - np.exp(-A * exp_kd_a * dt)
            
            # Apply boundary masks
            p_b[no_bid] = 0.0
            p_a[no_ask] = 0.0
            
            # Solve FOC for bid: uses p_a as p_other
            d_bid_new = solve_foc(Delta_b, p_a, Curv, no_bid)
            # Solve FOC for ask: uses p_b as p_other
            d_ask_new = solve_foc(Delta_a, p_b, Curv, no_ask)
            
            # Check convergence
            if np.max(np.abs(d_bid_new - d_bid)) < 1e-10 and \
               np.max(np.abs(d_ask_new - d_ask)) < 1e-10:
                d_bid = d_bid_new
                d_ask = d_ask_new
                break
            
            d_bid = d_bid_new
            d_ask = d_ask_new
        
        opt_bid[n, :] = d_bid
        opt_ask[n, :] = d_ask
        
        # Update value function
        exp_kd_b = np.exp(-k * d_bid)
        p_b = 1.0 - np.exp(-A * exp_kd_b * dt)
        exp_kd_a = np.exp(-k * d_ask)
        p_a = 1.0 - np.exp(-A * exp_kd_a * dt)
        p_b[no_bid] = 0.0
        p_a[no_ask] = 0.0
        
        V = p_b * (d_bid + Delta_b) + p_a * (d_ask + Delta_a) + V - cost - p_b * p_a * Curv

    # Ensure validity
    min_depth = -3.0 / k + 1e-10
    opt_bid = np.where(np.isfinite(opt_bid) & (opt_bid > min_depth), opt_bid, 1.0 / k)
    opt_ask = np.where(np.isfinite(opt_ask) & (opt_ask > min_depth), opt_ask, 1.0 / k)

    return opt_bid, opt_ask


def _get_policy(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_policy(params)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    steps = params['steps']
    Q = params['Q']

    try:
        opt_bid, opt_ask = _get_policy(params)
    except Exception:
        depth = np.full(len(inventory), 1.0 / params['k'])
        return depth, depth.copy()

    dt = params['T'] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    idx = np.clip(inventory + Q, 0, 2 * Q)

    bid_depths = opt_bid[n, idx]
    ask_depths = opt_ask[n, idx]

    k = params['k']
    min_depth = -3.0 / k + 1e-10
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_depth), bid_depths, 1.0 / k)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_depth), ask_depths, 1.0 / k)

    return bid_depths, ask_depths
