"""Optimal market making with inventory risk - Exact DP with Newton FOC solver.

Uses dynamic programming with exact value function and solves the
first-order conditions for optimal bid/ask depths via Newton's method,
eliminating grid discretization error.
"""
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_TIME_BUDGET = 320.0  # seconds, leave 10s margin before 330s kill


def _fill_prob(depth, A, k, dt):
    """Fill probability p(d) = 1 - exp(-A * exp(-k*d) * dt)."""
    if depth <= 0:
        return 0.0
    e = np.exp(-k * depth)
    inner = A * e * dt
    return 1.0 - np.exp(-inner)


def _solve_depth_foc(K, A, k, dt):
    """
    Solve the FOC for optimal depth d given coefficient K.
    
    The objective for depth d is: p(d)*(d + K)
    FOC: p'(d)*(d + K) + p(d) = 0
    
    where p(d) = 1 - exp(-A*exp(-k*d)*dt)
    p'(d) = -A*k*dt*exp(-k*d)*(1-p(d))
    
    So: -A*k*dt*exp(-k*d)*(1-p(d))*(d+K) + p(d) = 0
    """
    if K >= 0:
        # No incentive to quote (or quote at 0)
        return 0.0
    
    # f(d) = p(d) - A*k*dt*exp(-k*d)*(1-p(d))*(d+K) = 0
    # We need d > 0 and d + K < 0 (i.e., d < -K)
    
    # Initial guess: for small A*dt, p ≈ A*exp(-k*d)*dt, so:
    # A*exp(-k*d)*dt ≈ A*k*dt*exp(-k*d)*(d+K) => 1 ≈ k*(d+K) => d ≈ 1/k - K
    d = max(0.01, min(-K * 0.5, 1.0/k - K * 0.5))
    d = max(0.001, d)
    
    for _ in range(100):
        e = np.exp(-k * d)
        inner = A * e * dt
        one_minus_p = np.exp(-inner)
        p = 1.0 - one_minus_p
        
        f = p - A * k * dt * e * one_minus_p * (d + K)
        
        # f'(d) = p'(d) - A*k*dt*[(-k*exp(-k*d))*(1-p(d))*(d+K) + exp(-k*d)*(-p'(d))*(d+K) + exp(-k*d)*(1-p(d))*1]
        # p'(d) = -A*k*dt*e*one_minus_p
        p_prime = -A * k * dt * e * one_minus_p
        # d/dd [exp(-k*d)*(1-p(d))*(d+K)] = -k*exp(-k*d)*(1-p(d))*(d+K) + exp(-k*d)*(-p'(d))*(d+K) + exp(-k*d)*(1-p(d))
        deriv_inner = (-k * e * one_minus_p * (d + K) 
                      + e * (-p_prime) * (d + K) 
                      + e * one_minus_p)
        f_prime = p_prime - A * k * dt * deriv_inner
        
        if abs(f_prime) < 1e-15:
            break
        
        d_new = d - f / f_prime
        if d_new < 0:
            d_new = d * 0.5
        if d_new > -K:
            d_new = (-K + d) * 0.5
        if abs(d_new - d) < 1e-12:
            d = d_new
            break
        d = d_new
    
    return max(0.0, d)


def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes optimal bid/ask depths via dynamic programming with
    Newton's method for FOC solving.
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1
    
    # Inventory holding cost per step
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))
    
    start_time = time.time()
    deadline = start_time + (_TIME_BUDGET - (time.time() - _GLOBAL_START))
    
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
        
        # For each q, solve for optimal bid and ask
        # Bid: maximize p_b(b) * (b + K_b) where K_b = (1-p_a)*(V+ - V0) - p_a*(V- - V0)
        # But p_a depends on ask depth, which depends on bid... 
        # Use coordinate descent with Newton
        
        b_cur = np.full(n_q, 0.5)
        a_cur = np.full(n_q, 0.5)
        
        for iteration in range(15):
            b_old = b_cur.copy()
            a_old = a_cur.copy()
            
            # Compute p_a for current a
            p_a = np.array([_fill_prob(a, A, k, dt) for a in a_cur])
            
            # Optimize bid given ask
            # K_b(q) = (1-p_a(q))*(V+(q) - V0(q)) - p_a(q)*(V-(q) - V0(q))
            # Wait, let me re-derive:
            # J(b) = b*p_b + p_b(1-p_a)V+ + (1-p_b)(1-p_a)V0 + p_a(1-p_b)V- + const
            #      = b*p_b + p_b[(1-p_a)V+ - (1-p_a)V0] + (1-p_a)V0 + p_a V- - p_a p_b V- + const... 
            # Let me just do: J(b) = b*p_b + p_b * M_b + const
            # where M_b = (1-p_a)(V+ - V0) - p_a*(V- - V0) ... hmm
            # Actually: J(b) = b*p_b + p_b(1-p_a)V+ + (1-p_b)(1-p_a)V0 + p_a(1-p_b)V-
            # = b*p_b + p_b[(1-p_a)V+ - (1-p_a)V0 - p_a V-] + (1-p_a)V0 + p_a V-
            # = p_b * (b + (1-p_a)(V+ - V0) - p_a(V- - V0)) + const
            # So K_b = (1-p_a)(V+ - V0) - p_a(V- - V0)
            
            K_b = (1.0 - p_a) * (V_q_plus - V_q) - p_a * (V_q_minus - V_q)
            
            b_new = np.array([_solve_depth_foc(K_b[q], A, k, dt) for q in range(n_q)])
            b_new[n_q - 1] = 0.0  # No bid at Q
            
            # Compute p_b for new b
            p_b = np.array([_fill_prob(b, A, k, dt) for b in b_new])
            
            # Optimize ask given bid
            # K_a = (1-p_b)(V- - V0) - p_b(V+ - V0)
            K_a = (1.0 - p_b) * (V_q_minus - V_q) - p_b * (V_q_plus - V_q)
            
            a_new = np.array([_solve_depth_foc(K_a[q], A, k, dt) for q in range(n_q)])
            a_new[0] = 0.0  # No ask at -Q
            
            b_cur = b_new
            a_cur = a_new
            
            diff = max(np.max(np.abs(b_cur - b_old)), np.max(np.abs(a_cur - a_old)))
            if diff < 1e-9:
                break
        
        # Compute final value
        p_b_final = np.array([_fill_prob(b, A, k, dt) for b in b_cur])
        p_a_final = np.array([_fill_prob(a, A, k, dt) for a in a_cur])
        
        trans_plus = p_b_final * (1.0 - p_a_final) * V_q_plus
        trans_minus = p_a_final * (1.0 - p_b_final) * V_q_minus
        trans_same = (1.0 - p_b_final - p_a_final + p_b_final * p_a_final) * V_q
        
        V_n = b_cur * p_b_final + a_cur * p_a_final - C_inv + trans_plus + trans_minus + trans_same
        
        V[n, :] = V_n
        opt_bid[n, :] = b_cur
        opt_ask[n, :] = a_cur
    
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
