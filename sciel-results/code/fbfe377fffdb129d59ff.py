"""Optimal market making with inventory risk - Vectorized Grid Search Solver.

Uses dynamic programming with a fine grid of candidate depths and
vectorized coordinate descent to find optimal bid/ask spreads.
"""
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_TIME_BUDGET = 320.0  # seconds, leave 10s margin before 330s kill


def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes optimal bid/ask depths via dynamic programming with
    vectorized grid search and coordinate descent.
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
    
    # Define a fine grid for depths
    # Estimate max depth needed
    # p < 1e-6 => exp(-k d) < 1e-6 / (A dt)
    # d > -1/k * ln(1e-6 / (A dt))
    max_depth = 20.0  # Safe upper bound for all regimes
    n_grid = 5000
    depths = np.linspace(0, max_depth, n_grid)
    
    # Pre-compute fill probabilities for the grid
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    exp_term = np.exp(-k * depths)
    inner = A * exp_term * dt
    p_grid = 1.0 - np.exp(-inner)
    
    # Initial guesses
    b_cur = np.full(n_q, 0.5)
    a_cur = np.full(n_q, 0.5)
    
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
        
        # Coordinate descent iterations
        # We alternate optimizing b and a.
        # Since the problem is convex-like (unimodal objective), this converges quickly.
        max_iters = 20
        tol = 1e-8
        
        for _ in range(max_iters):
            b_old = b_cur.copy()
            a_old = a_cur.copy()
            
            # 1. Optimize bid (b) given ask (a)
            # Compute p_a for current a
            # a_cur is shape (n_q,)
            # We need p_a for each q.
            # Since a_cur might not be on the grid, we compute p directly.
            exp_a = np.exp(-k * a_cur)
            p_a = 1.0 - np.exp(-A * exp_a * dt)
            
            # Coefficient K_q for bid optimization
            # J_q(b) = p_b(b) * (b + K_q) + const
            # K_q = (1 - p_a) * (V_plus - V_same) - p_a * V_minus
            # Note: For q=Q, bid is irrelevant (no fill), but we can still compute.
            # For q=-Q, ask is irrelevant.
            
            K = (1.0 - p_a) * (V_q_plus - V_q) - p_a * V_q_minus
            
            # Compute scores for all grid points b for all q
            # p_grid is (n_grid,)
            # depths is (n_grid,)
            # K is (n_q,)
            # We want to maximize p_grid[:, None] * (depths[:, None] + K[None, :])
            # Shape: (n_grid, n_q)
            
            # Optimization: We can avoid the full broadcast if we are smart, 
            # but 5000x61 is small enough.
            scores_b = p_grid[:, None] * (depths[:, None] + K[None, :])
            
            # Find best b for each q
            best_idx_b = np.argmax(scores_b, axis=0)
            b_new = depths[best_idx_b]
            
            # Handle boundary conditions:
            # At q=Q (index n_q-1), bid fill is impossible.
            # The value function doesn't depend on b.
            # We can set b to 0 or any value. Let's set to 0 to minimize "spread" if needed, 
            # but it doesn't affect value.
            b_new[n_q - 1] = 0.0
            
            # 2. Optimize ask (a) given bid (b_new)
            exp_b = np.exp(-k * b_new)
            p_b = 1.0 - np.exp(-A * exp_b * dt)
            
            # Coefficient L_q for ask optimization
            # J_q(a) = p_a(a) * (a + L_q) + const
            # L_q = (1 - p_b) * (V_minus - V_same) - p_b * V_plus
            
            L = (1.0 - p_b) * (V_q_minus - V_q) - p_b * V_q_plus
            
            scores_a = p_grid[:, None] * (depths[:, None] + L[None, :])
            
            best_idx_a = np.argmax(scores_a, axis=0)
            a_new = depths[best_idx_a]
            
            # At q=-Q (index 0), ask fill is impossible.
            a_new[0] = 0.0
            
            # Check convergence
            diff_b = np.max(np.abs(b_new - b_old))
            diff_a = np.max(np.abs(a_new - a_old))
            
            b_cur = b_new
            a_cur = a_new
            
            if diff_b < tol and diff_a < tol:
                break
        
        # Compute final value for this step
        # p_b, p_a for final b_cur, a_cur
        exp_b_final = np.exp(-k * b_cur)
        p_b_final = 1.0 - np.exp(-A * exp_b_final * dt)
        
        exp_a_final = np.exp(-k * a_cur)
        p_a_final = 1.0 - np.exp(-A * exp_a_final * dt)
        
        # Calculate V[n, q]
        # V = b*p_b + a*p_a - C_inv + p_b(1-p_a)V+ + p_a(1-p_b)V- + (1-p_b-p_a+p_b p_a)V0
        
        trans_plus = p_b_final * (1.0 - p_a_final) * V_q_plus
        trans_minus = p_a_final * (1.0 - p_b_final) * V_q_minus
        trans_same = (1.0 - p_b_final - p_a_final + p_b_final * p_a_final) * V_q
        
        V_n = b_cur * p_b_final + a_cur * p_a_final - C_inv + trans_plus + trans_minus + trans_same
        
        # Boundary corrections for value (though b/a=0 handles it mostly)
        # If q=Q, b=0, p_b=0. V = a*p_a - C + p_a V- + (1-p_a) V0. Correct.
        # If q=-Q, a=0, p_a=0. V = b*p_b - C + p_b V+ + (1-p_b) V0. Correct.
        
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
