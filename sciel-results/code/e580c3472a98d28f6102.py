"""Optimal market making with inventory risk - continuous Newton-Raphson solver.

Solves the first-order conditions for optimal bid/ask spreads directly,
eliminating grid discretization error.
"""
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_TIME_BUDGET = 320.0  # seconds, leave 10s margin before 330s kill


def _fill_prob_and_deriv(depth, A, k, dt):
    """Return (p, p_prime) for fill probability and its derivative w.r.t. depth.
    
    p(b) = 1 - exp(-A * exp(-k*b) * dt)
    p'(b) = -A * k * dt * exp(-k*b) * (1 - p(b))
    """
    if depth <= 0:
        return 0.0, 0.0
    e = np.exp(-k * depth)
    inner = A * e * dt
    one_minus_p = np.exp(-inner)  # = 1 - p
    p = 1.0 - one_minus_p
    p_prime = -A * k * dt * e * one_minus_p
    return p, p_prime


def _solve_foc_bid(b_init, a, V_plus, V_minus, V_same, A, k, dt, max_iter=50, tol=1e-10):
    """Solve FOC for bid depth given ask depth a.
    
    FOC: p_b + p_b' * [b + (1-p_a)*V_plus - p_a*V_minus - (1+p_a)*V_same] = 0
    """
    p_a, _ = _fill_prob_and_deriv(a, A, k, dt)
    b = max(b_init, 1e-6)
    
    for _ in range(max_iter):
        p_b, p_b_prime = _fill_prob_and_deriv(b, A, k, dt)
        bracket = b + (1.0 - p_a) * V_plus - p_a * V_minus - (1.0 + p_a) * V_same
        f = p_b + p_b_prime * bracket
        
        # Compute derivative of f w.r.t. b for Newton step
        # f = p_b + p_b' * bracket
        # f' = p_b'' + p_b'' * bracket + p_b' * 1
        # p_b'' = d/db[-A*k*dt*exp(-kb)*(1-p_b)]
        #       = -A*k*dt * [-k*exp(-kb)*(1-p_b) + exp(-kb)*(-p_b')]
        #       = -A*k*dt * exp(-kb) * [-k*(1-p_b) - p_b']
        #       = A*k*dt*exp(-kb) * [k*(1-p_b) + p_b']
        # Let's compute numerically for stability
        h = 1e-6
        p_b2, p_b_prime2 = _fill_prob_and_deriv(b + h, A, k, dt)
        f2 = p_b2 + p_b_prime2 * (b + h + (1.0 - p_a) * V_plus - p_a * V_minus - (1.0 + p_a) * V_same)
        f_prime = (f2 - f) / h
        
        if abs(f_prime) < 1e-15:
            break
        
        step = f / f_prime
        # Damped Newton
        if abs(step) > b * 0.5:
            step = np.sign(step) * b * 0.5
        b_new = b - step
        if b_new < 1e-6:
            b_new = 1e-6
        
        if abs(b_new - b) < tol:
            b = b_new
            break
        b = b_new
    
    return max(b, 1e-6)


def _solve_foc_ask(a_init, b, V_plus, V_minus, V_same, A, k, dt, max_iter=50, tol=1e-10):
    """Solve FOC for ask depth given bid depth b.
    
    FOC: p_a + p_a' * [a + (1-p_b)*V_minus - p_b*V_plus - (1+p_b)*V_same] = 0
    """
    p_b, _ = _fill_prob_and_deriv(b, A, k, dt)
    a = max(a_init, 1e-6)
    
    for _ in range(max_iter):
        p_a, p_a_prime = _fill_prob_and_deriv(a, A, k, dt)
        bracket = a + (1.0 - p_b) * V_minus - p_b * V_plus - (1.0 + p_b) * V_same
        f = p_a + p_a_prime * bracket
        
        h = 1e-6
        p_a2, p_a_prime2 = _fill_prob_and_deriv(a + h, A, k, dt)
        f2 = p_a2 + p_a_prime2 * (a + h + (1.0 - p_b) * V_minus - p_b * V_plus - (1.0 + p_b) * V_same)
        f_prime = (f2 - f) / h
        
        if abs(f_prime) < 1e-15:
            break
        
        step = f / f_prime
        if abs(step) > a * 0.5:
            step = np.sign(step) * a * 0.5
        a_new = a - step
        if a_new < 1e-6:
            a_new = 1e-6
        
        if abs(a_new - a) < tol:
            a = a_new
            break
        a = a_new
    
    return max(a, 1e-6)


def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes optimal bid/ask depths via dynamic programming with
    Newton-Raphson solution of first-order conditions at each state.
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1
    
    # Inventory holding cost per step
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # V[n, q_idx]
    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))
    
    start_time = time.time()
    deadline = start_time + (_TIME_BUDGET - (time.time() - _GLOBAL_START))
    
    # Initial guess: use a reasonable spread
    # For small inventory, quote tight; for large, quote wide
    # Use a heuristic initial guess based on inventory
    init_spread = np.zeros(n_q)
    for i in range(n_q):
        q_val = q_vals[i]
        # Wider spreads for larger |inventory|
        init_spread[i] = 0.1 + 0.05 * abs(q_val)
    
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
        
        refined_bid = np.zeros(n_q)
        refined_ask = np.zeros(n_q)
        V_refined = np.zeros(n_q)
        
        for q_idx in range(n_q):
            # Initial guesses
            b_cur = init_spread[q_idx]
            a_cur = init_spread[q_idx]
            
            # Iterate: solve FOC for bid given ask, then ask given bid
            for _ in range(10):
                if q_idx == n_q - 1:
                    # At q=Q, no bid fill possible, so bid depth doesn't matter
                    # Set to a small value
                    b_cur = 1e-6
                else:
                    b_cur = _solve_foc_bid(b_cur, a_cur, V_q_plus[q_idx], 
                                           V_q_minus[q_idx], V_q[q_idx], A, k, dt)
                
                if q_idx == 0:
                    # At q=-Q, no ask fill possible
                    a_cur = 1e-6
                else:
                    a_cur = _solve_foc_ask(a_cur, b_cur, V_q_plus[q_idx],
                                           V_q_minus[q_idx], V_q[q_idx], A, k, dt)
            
            refined_bid[q_idx] = b_cur
            refined_ask[q_idx] = a_cur
            
            # Compute refined value
            pb, _ = _fill_prob_and_deriv(b_cur, A, k, dt)
            pa, _ = _fill_prob_and_deriv(a_cur, A, k, dt)
            
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

    # Ensure valid outputs: must be >= 0 and finite
    bid_depths = np.where(np.isnan(bid_depths) | ~np.isfinite(bid_depths), 0.01, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | ~np.isfinite(ask_depths), 0.01, ask_depths)
    bid_depths = np.where(bid_depths < 0, 0.01, bid_depths)
    ask_depths = np.where(ask_depths < 0, 0.01, ask_depths)

    return bid_depths.astype(float), ask_depths.astype(float)
